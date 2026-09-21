//go:build !unix

package cligate

import "os/exec"

func killGroup(*exec.Cmd) {}
