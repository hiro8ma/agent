package chapter02

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

const (
	CalendarReadonlyScope    = "https://www.googleapis.com/auth/calendar.readonly"
	CalendarCredentialRefKey = "calendar_credential_ref"
)

var (
	ErrCalendarAuthenticationRequired = errors.New("Google Calendar の認証が必要")
	ErrCalendarForbidden              = errors.New("Google Calendar を読む権限がない")
)

type AuthorizationRequest struct {
	URL          string
	CodeVerifier string
}

func NewGoogleCalendarOAuthConfig(
	clientID string,
	clientSecret string,
	redirectURL string,
) (*oauth2.Config, error) {
	if strings.TrimSpace(clientID) == "" || strings.TrimSpace(clientSecret) == "" {
		return nil, errors.New("OAuth のクライアント ID とクライアントシークレットが必要")
	}
	if _, err := url.ParseRequestURI(redirectURL); err != nil {
		return nil, fmt.Errorf("OAuth のリダイレクト URL: %w", err)
	}
	return &oauth2.Config{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURL:  redirectURL,
		Scopes:       []string{CalendarReadonlyScope},
		Endpoint:     google.Endpoint,
	}, nil
}

func NewAuthorizationRequest(config *oauth2.Config, state string) (AuthorizationRequest, error) {
	if config == nil {
		return AuthorizationRequest{}, errors.New("OAuth の設定が必要")
	}
	if strings.TrimSpace(state) == "" {
		return AuthorizationRequest{}, errors.New("OAuth の state が必要")
	}
	verifier := oauth2.GenerateVerifier()
	return AuthorizationRequest{
		URL: config.AuthCodeURL(
			state,
			oauth2.AccessTypeOffline,
			oauth2.S256ChallengeOption(verifier),
		),
		CodeVerifier: verifier,
	}, nil
}

func ExchangeAuthorizationCode(
	ctx context.Context,
	config *oauth2.Config,
	code string,
	codeVerifier string,
) (*oauth2.Token, error) {
	if config == nil {
		return nil, errors.New("OAuth の設定が必要")
	}
	if strings.TrimSpace(code) == "" || strings.TrimSpace(codeVerifier) == "" {
		return nil, errors.New("認可コードと PKCE の verifier が必要")
	}
	token, err := config.Exchange(ctx, code, oauth2.VerifierOption(codeVerifier))
	if err != nil {
		return nil, fmt.Errorf("OAuth の認可コード交換: %w", err)
	}
	return token, nil
}

type CalendarEvent struct {
	ID      string         `json:"id"`
	Summary string         `json:"summary"`
	Start   map[string]any `json:"start"`
	End     map[string]any `json:"end"`
}

type CalendarClient struct {
	HTTPClient *http.Client
	BaseURL    string
}

func (c *CalendarClient) ListEvents(
	ctx context.Context,
	targetDate string,
	timeZone string,
	accessToken string,
) ([]CalendarEvent, error) {
	if strings.TrimSpace(timeZone) == "" {
		timeZone = "Asia/Tokyo"
	}
	location, err := time.LoadLocation(timeZone)
	if err != nil {
		return nil, fmt.Errorf("IANA 形式のタイムゾーンを指定する: %w", err)
	}
	day, err := time.ParseInLocation(time.DateOnly, targetDate, location)
	if err != nil {
		return nil, fmt.Errorf("日付は YYYY-MM-DD で指定する: %w", err)
	}
	if strings.TrimSpace(accessToken) == "" {
		return nil, ErrCalendarAuthenticationRequired
	}

	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
	}
	values := url.Values{
		"timeMin":      {day.Format(time.RFC3339)},
		"timeMax":      {day.AddDate(0, 0, 1).Format(time.RFC3339)},
		"singleEvents": {"true"},
		"orderBy":      {"startTime"},
		"timeZone":     {timeZone},
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"?"+values.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("Calendar API のリクエスト作成: %w", err)
	}
	request.Header.Set("Authorization", "Bearer "+accessToken)

	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("Calendar API の呼び出し: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode == http.StatusUnauthorized {
		return nil, ErrCalendarAuthenticationRequired
	}
	if response.StatusCode == http.StatusForbidden {
		return nil, ErrCalendarForbidden
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("Calendar API が HTTP %d を返した", response.StatusCode)
	}

	var payload struct {
		Items []CalendarEvent `json:"items"`
	}
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("Calendar API の応答を読む: %w", err)
	}
	return payload.Items, nil
}

type AccessTokenResolver interface {
	ResolveAccessToken(context.Context, string) (string, error)
}

type GetCalendarEventsInput struct {
	Date     string `json:"date"`
	TimeZone string `json:"timeZone,omitempty"`
}

type GetCalendarEventsOutput struct {
	Status     string          `json:"status"`
	EventCount int             `json:"eventCount,omitempty"`
	Events     []CalendarEvent `json:"events,omitempty"`
	Message    string          `json:"message,omitempty"`
}

func NewCalendarEventsTool(
	client *CalendarClient,
	tokens AccessTokenResolver,
) (tool.Tool, error) {
	if client == nil || tokens == nil {
		return nil, errors.New("Calendar API クライアントとトークン解決器が必要")
	}
	return functiontool.New(functiontool.Config{
		Name:        "get_calendar_events",
		Description: "認証済みの利用者について、指定日の Google Calendar の予定を読む。",
	}, func(ctx agent.Context, input GetCalendarEventsInput) (GetCalendarEventsOutput, error) {
		value, err := ctx.State().Get(CalendarCredentialRefKey)
		if err != nil {
			return GetCalendarEventsOutput{
				Status:  "auth_required",
				Message: "Google Calendar へのアクセスを許可してください。",
			}, nil
		}
		credentialRef, ok := value.(string)
		if !ok || strings.TrimSpace(credentialRef) == "" {
			return GetCalendarEventsOutput{
				Status:  "auth_required",
				Message: "Google Calendar の認証情報を確認できませんでした。",
			}, nil
		}

		accessToken, err := tokens.ResolveAccessToken(ctx, credentialRef)
		if err != nil {
			if !errors.Is(err, ErrCalendarAuthenticationRequired) {
				return GetCalendarEventsOutput{
					Status:  "error",
					Message: "認証情報ストアから資格情報を取得できませんでした。",
				}, nil
			}
			return GetCalendarEventsOutput{
				Status:  "auth_required",
				Message: "Google Calendar の認証を更新してください。",
			}, nil
		}
		events, err := client.ListEvents(ctx, input.Date, input.TimeZone, accessToken)
		if errors.Is(err, ErrCalendarAuthenticationRequired) {
			return GetCalendarEventsOutput{
				Status:  "auth_required",
				Message: "Google Calendar の認証を更新してください。",
			}, nil
		}
		if errors.Is(err, ErrCalendarForbidden) {
			return GetCalendarEventsOutput{
				Status:  "forbidden",
				Message: "Google Calendar を読む権限がありません。",
			}, nil
		}
		if err != nil {
			return GetCalendarEventsOutput{
				Status:  "error",
				Message: err.Error(),
			}, nil
		}
		return GetCalendarEventsOutput{
			Status: "success", EventCount: len(events), Events: events,
		}, nil
	})
}
