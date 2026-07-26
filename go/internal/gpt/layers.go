package gpt

import "math"

// B=バッチ, T=系列長, C=埋め込み次元, NH=ヘッド数, HS=C/NH, V=語彙数, OC=出力次元。
// matmul の重みは [OC, C] レイアウトで out = inp @ W^T + b。

// encoderForward: out[b,t,:] = wte[token] + wpe[t]
func encoderForward(out []float64, idx []int, wte, wpe []float64, B, T, C int) {
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			o := (b*T + t) * C
			tok := idx[b*T+t] * C
			pos := t * C
			for c := 0; c < C; c++ {
				out[o+c] = wte[tok+c] + wpe[pos+c]
			}
		}
	}
}

func encoderBackward(dwte, dwpe []float64, dout []float64, idx []int, B, T, C int) {
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			o := (b*T + t) * C
			tok := idx[b*T+t] * C
			pos := t * C
			for c := 0; c < C; c++ {
				d := dout[o+c]
				dwte[tok+c] += d
				dwpe[pos+c] += d
			}
		}
	}
}

// mean と rstd は backward で再利用するためキャッシュする。
func layernormForward(out, mean, rstd, inp, weight, bias []float64, N, C int) {
	const eps = 1e-5
	for n := 0; n < N; n++ {
		x := inp[n*C : (n+1)*C]
		m := 0.0
		for c := 0; c < C; c++ {
			m += x[c]
		}
		m /= float64(C)
		v := 0.0
		for c := 0; c < C; c++ {
			d := x[c] - m
			v += d * d
		}
		v /= float64(C)
		s := 1.0 / math.Sqrt(v+eps)
		o := out[n*C : (n+1)*C]
		for c := 0; c < C; c++ {
			o[c] = (x[c]-m)*s*weight[c] + bias[c]
		}
		mean[n] = m
		rstd[n] = s
	}
}

func layernormBackward(dinp, dweight, dbias, dout, inp, weight, mean, rstd []float64, N, C int) {
	for n := 0; n < N; n++ {
		x := inp[n*C : (n+1)*C]
		do := dout[n*C : (n+1)*C]
		di := dinp[n*C : (n+1)*C]
		m, s := mean[n], rstd[n]

		dnormMean := 0.0
		dnormNormMean := 0.0
		for c := 0; c < C; c++ {
			norm := (x[c] - m) * s
			dnorm := weight[c] * do[c]
			dnormMean += dnorm
			dnormNormMean += dnorm * norm
		}
		dnormMean /= float64(C)
		dnormNormMean /= float64(C)

		for c := 0; c < C; c++ {
			norm := (x[c] - m) * s
			dnorm := weight[c] * do[c]
			dbias[c] += do[c]
			dweight[c] += norm * do[c]
			di[c] += (dnorm - dnormMean - norm*dnormNormMean) * s
		}
	}
}

// matmulForward: out[n,o] = bias[o] + Σ_c inp[n,c] * weight[o*C+c]
func matmulForward(out, inp, weight, bias []float64, N, C, OC int) {
	for n := 0; n < N; n++ {
		x := inp[n*C : (n+1)*C]
		o := out[n*OC : (n+1)*OC]
		for oc := 0; oc < OC; oc++ {
			val := 0.0
			if bias != nil {
				val = bias[oc]
			}
			w := weight[oc*C : (oc+1)*C]
			for c := 0; c < C; c++ {
				val += x[c] * w[c]
			}
			o[oc] = val
		}
	}
}

func matmulBackward(dinp, dweight, dbias, dout, inp, weight []float64, N, C, OC int) {
	for n := 0; n < N; n++ {
		do := dout[n*OC : (n+1)*OC]
		di := dinp[n*C : (n+1)*C]
		x := inp[n*C : (n+1)*C]
		for oc := 0; oc < OC; oc++ {
			d := do[oc]
			if dbias != nil {
				dbias[oc] += d
			}
			w := weight[oc*C : (oc+1)*C]
			dw := dweight[oc*C : (oc+1)*C]
			for c := 0; c < C; c++ {
				di[c] += w[c] * d
				dw[c] += x[c] * d
			}
		}
	}
}

// causal 自己アテンション。qkv は [B,T,3C]（q,k,v 連結）、att は backward 用にキャッシュ。
func attentionForward(out, preatt, att, qkv []float64, B, T, C, NH int) {
	HS := C / NH
	scale := 1.0 / math.Sqrt(float64(HS))
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < NH; h++ {
				q := qkv[(b*T+t)*3*C+h*HS:]
				pre := preatt[((b*NH+h)*T+t)*T:]
				at := att[((b*NH+h)*T+t)*T:]

				maxv := math.Inf(-1)
				for t2 := 0; t2 <= t; t2++ {
					k := qkv[(b*T+t2)*3*C+C+h*HS:]
					val := 0.0
					for i := 0; i < HS; i++ {
						val += q[i] * k[i]
					}
					val *= scale
					pre[t2] = val
					if val > maxv {
						maxv = val
					}
				}
				sum := 0.0
				for t2 := 0; t2 <= t; t2++ {
					e := math.Exp(pre[t2] - maxv)
					at[t2] = e
					sum += e
				}
				inv := 1.0 / sum
				for t2 := 0; t2 <= t; t2++ {
					at[t2] *= inv
				}
				for t2 := t + 1; t2 < T; t2++ {
					at[t2] = 0
				}
				o := out[(b*T+t)*C+h*HS:]
				for i := 0; i < HS; i++ {
					o[i] = 0
				}
				for t2 := 0; t2 <= t; t2++ {
					v := qkv[(b*T+t2)*3*C+2*C+h*HS:]
					a := at[t2]
					for i := 0; i < HS; i++ {
						o[i] += a * v[i]
					}
				}
			}
		}
	}
}

func attentionBackward(dqkv, dpreatt, datt, dout, qkv, att []float64, B, T, C, NH int) {
	HS := C / NH
	scale := 1.0 / math.Sqrt(float64(HS))
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < NH; h++ {
				at := att[((b*NH+h)*T+t)*T:]
				dat := datt[((b*NH+h)*T+t)*T:]
				dpre := dpreatt[((b*NH+h)*T+t)*T:]
				do := dout[(b*T+t)*C+h*HS:]
				q := qkv[(b*T+t)*3*C+h*HS:]
				dq := dqkv[(b*T+t)*3*C+h*HS:]

				for t2 := 0; t2 <= t; t2++ {
					v := qkv[(b*T+t2)*3*C+2*C+h*HS:]
					dv := dqkv[(b*T+t2)*3*C+2*C+h*HS:]
					for i := 0; i < HS; i++ {
						dat[t2] += v[i] * do[i]
						dv[i] += at[t2] * do[i]
					}
				}
				for t2 := 0; t2 <= t; t2++ {
					for t3 := 0; t3 <= t; t3++ {
						ind := 0.0
						if t2 == t3 {
							ind = 1.0
						}
						dpre[t3] += at[t2] * (ind - at[t3]) * dat[t2]
					}
				}
				for t2 := 0; t2 <= t; t2++ {
					k := qkv[(b*T+t2)*3*C+C+h*HS:]
					dk := dqkv[(b*T+t2)*3*C+C+h*HS:]
					d := dpre[t2] * scale
					for i := 0; i < HS; i++ {
						dq[i] += k[i] * d
						dk[i] += q[i] * d
					}
				}
			}
		}
	}
}

const geluScale = 0.7978845608028654 // sqrt(2/pi)

// tanh 近似 GELU。
func geluForward(out, inp []float64) {
	for i, x := range inp {
		cube := 0.044715 * x * x * x
		out[i] = 0.5 * x * (1.0 + math.Tanh(geluScale*(x+cube)))
	}
}

func geluBackward(dinp, inp, dout []float64) {
	for i, x := range inp {
		cube := 0.044715 * x * x * x
		arg := geluScale * (x + cube)
		tanhOut := math.Tanh(arg)
		sech2 := 1.0 - tanhOut*tanhOut
		local := 0.5*(1.0+tanhOut) + x*0.5*sech2*geluScale*(1.0+3.0*0.044715*x*x)
		dinp[i] += local * dout[i]
	}
}

func residualForward(out, a, b []float64) {
	for i := range out {
		out[i] = a[i] + b[i]
	}
}

func residualBackward(da, db, dout []float64) {
	for i := range dout {
		da[i] += dout[i]
		db[i] += dout[i]
	}
}

// softmaxForward: logits [N,V] -> probs [N,V]
func softmaxForward(probs, logits []float64, N, V int) {
	for n := 0; n < N; n++ {
		l := logits[n*V : (n+1)*V]
		p := probs[n*V : (n+1)*V]
		maxv := math.Inf(-1)
		for _, v := range l {
			if v > maxv {
				maxv = v
			}
		}
		sum := 0.0
		for i, v := range l {
			e := math.Exp(v - maxv)
			p[i] = e
			sum += e
		}
		inv := 1.0 / sum
		for i := range p {
			p[i] *= inv
		}
	}
}

// crossentropyForward: losses[n] = -log(probs[n, target[n]])
func crossentropyForward(losses, probs []float64, targets []int, N, V int) {
	for n := 0; n < N; n++ {
		losses[n] = -math.Log(probs[n*V+targets[n]])
	}
}

// crossentropySoftmaxBackward: dlogits = (probs - onehot) * dloss
func crossentropySoftmaxBackward(dlogits, probs []float64, targets []int, N, V int, dloss float64) {
	for n := 0; n < N; n++ {
		p := probs[n*V : (n+1)*V]
		dl := dlogits[n*V : (n+1)*V]
		tgt := targets[n]
		for i := 0; i < V; i++ {
			ind := 0.0
			if i == tgt {
				ind = 1.0
			}
			dl[i] += (p[i] - ind) * dloss
		}
	}
}
