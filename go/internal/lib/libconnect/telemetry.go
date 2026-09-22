package libconnect

import (
	"connectrpc.com/connect"
	"connectrpc.com/otelconnect"
)

// Telemetry は内部のサービスの呼び出しを span とメトリクスにする。受けた trace の子として span を作る。
func Telemetry() connect.Interceptor {
	return newTelemetry(otelconnect.WithTrustRemote())
}

// EdgeTelemetry は外から呼ばれる口に使う。受けた trace は信用せず、新しい trace を始めてリンクだけ残す。
func EdgeTelemetry() connect.Interceptor {
	return newTelemetry()
}

func newTelemetry(opts ...otelconnect.Option) connect.Interceptor {
	opts = append(opts, otelconnect.WithoutServerPeerAttributes())
	i, err := otelconnect.NewInterceptor(opts...)
	if err != nil {
		// メトリクスの計器を作れないときだけ失敗するので、trace だけは残す。
		i, _ = otelconnect.NewInterceptor(append(opts, otelconnect.WithoutMetrics())...)
	}
	return i
}
