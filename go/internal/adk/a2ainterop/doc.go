// Package a2ainterop は、ADK のエージェントを A2A のサーバーとして公開したときの、プロトコルの版の食い違いを確かめる。
//
// Go の ADK v2.2.0 は A2A を 2 つの版で持つ。server/adka2a は a2a-go v0.3（プロトコル 0.3）、
// server/adka2a/v2 は a2a-go/v2（プロトコル 1.0）。Python の ADK v2.2.0 は a2a-sdk 0.3 系に固定されている。
// v1.0 のサーバーは既定では v0.3 のリクエストを受けないので、Python の ADK から呼ばせるには互換の口（a2av0）を足す。
package a2ainterop
