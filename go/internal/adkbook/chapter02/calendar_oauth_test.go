package chapter02

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"golang.org/x/oauth2"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestNewAuthorizationRequestUsesStateAndPKCE(t *testing.T) {
	config := &oauth2.Config{
		ClientID:    "client-id",
		RedirectURL: "http://127.0.0.1/callback",
		Scopes:      []string{CalendarReadonlyScope},
		Endpoint: oauth2.Endpoint{
			AuthURL: "https://accounts.example.test/authorize",
		},
	}

	request, err := NewAuthorizationRequest(config, "session-state")
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := url.Parse(request.URL)
	if err != nil {
		t.Fatal(err)
	}
	query := parsed.Query()
	if query.Get("state") != "session-state" {
		t.Fatalf("state = %q", query.Get("state"))
	}
	if query.Get("code_challenge_method") != "S256" || query.Get("code_challenge") == "" {
		t.Fatalf("PKCE が無い: %s", request.URL)
	}
	if request.CodeVerifier == "" {
		t.Fatal("verifier が空")
	}
	if query.Get("access_type") != "offline" {
		t.Fatalf("access_type = %q", query.Get("access_type"))
	}
}

func TestExchangeAuthorizationCodeSendsVerifier(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		if r.Form.Get("code") != "authorization-code" {
			t.Errorf("code = %q", r.Form.Get("code"))
		}
		if r.Form.Get("code_verifier") != "verifier" {
			t.Errorf("code_verifier = %q", r.Form.Get("code_verifier"))
		}
		body, err := json.Marshal(map[string]any{
			"access_token":  "access-token",
			"refresh_token": "refresh-token",
			"token_type":    "Bearer",
			"expires_in":    3600,
		})
		if err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(string(body))),
			Request:    r,
		}, nil
	})}

	config := &oauth2.Config{
		ClientID:     "client-id",
		ClientSecret: "client-secret",
		RedirectURL:  "http://127.0.0.1/callback",
		Endpoint:     oauth2.Endpoint{TokenURL: "https://oauth.example.test/token"},
	}
	ctx := context.WithValue(context.Background(), oauth2.HTTPClient, client)
	token, err := ExchangeAuthorizationCode(
		ctx, config, "authorization-code", "verifier",
	)
	if err != nil {
		t.Fatal(err)
	}
	if token.AccessToken != "access-token" || token.RefreshToken != "refresh-token" {
		t.Fatalf("token = %#v", token)
	}
}

func TestCalendarClientUsesBearerTokenAndDayRange(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Header.Get("Authorization") != "Bearer access-token" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		if r.URL.Query().Get("timeMin") != "2026-09-10T00:00:00+09:00" {
			t.Errorf("timeMin = %q", r.URL.Query().Get("timeMin"))
		}
		if r.URL.Query().Get("timeMax") != "2026-09-11T00:00:00+09:00" {
			t.Errorf("timeMax = %q", r.URL.Query().Get("timeMax"))
		}
		if r.URL.Query().Get("timeZone") != "Asia/Tokyo" {
			t.Errorf("timeZone = %q", r.URL.Query().Get("timeZone"))
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"items":[{"id":"event-1","summary":"予定"}]}`,
			)),
			Request: r,
		}, nil
	})}

	client := CalendarClient{
		HTTPClient: httpClient,
		BaseURL:    "https://calendar.example.test/events",
	}
	events, err := client.ListEvents(
		context.Background(), "2026-09-10", "Asia/Tokyo", "access-token",
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].ID != "event-1" {
		t.Fatalf("events = %#v", events)
	}
}

func TestCalendarClientDoesNotExposeResponseBodyInError(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       io.NopCloser(strings.NewReader("secret-token-value")),
			Request:    r,
		}, nil
	})}

	client := CalendarClient{
		HTTPClient: httpClient,
		BaseURL:    "https://calendar.example.test/events",
	}
	_, err := client.ListEvents(
		context.Background(), "2026-09-10", "Asia/Tokyo", "access-token",
	)
	if err == nil {
		t.Fatal("エラーにならなかった")
	}
	if strings.Contains(err.Error(), "secret-token-value") {
		t.Fatalf("応答本文がエラーへ漏れた: %v", err)
	}
}

func TestCalendarClientSeparatesUnauthorizedAndForbidden(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		want       error
	}{
		{name: "unauthorized", statusCode: http.StatusUnauthorized, want: ErrCalendarAuthenticationRequired},
		{name: "forbidden", statusCode: http.StatusForbidden, want: ErrCalendarForbidden},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: test.statusCode,
					Body:       io.NopCloser(strings.NewReader("")),
					Request:    r,
				}, nil
			})}
			client := CalendarClient{
				HTTPClient: httpClient,
				BaseURL:    "https://calendar.example.test/events",
			}
			_, err := client.ListEvents(
				context.Background(), "2026-09-10", "Asia/Tokyo", "access-token",
			)
			if !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestBudgetOutputSchemaFixesFinalShape(t *testing.T) {
	schema := budgetOutputSchema()
	for _, name := range []string{
		"currency",
		"transportation_yen",
		"food_yen",
		"activities_yen",
		"total_yen",
		"assumptions",
	} {
		if schema.Properties[name] == nil {
			t.Fatalf("%s が schema に無い", name)
		}
	}
	if len(schema.Required) != len(schema.Properties) {
		t.Fatalf("required = %v", schema.Required)
	}
}
