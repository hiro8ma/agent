//go:build unix

package cligate

import (
	"os/exec"
	"syscall"
)

// killGroup は子プロセスを新しいプロセスグループで起動し、止めるときはグループごと止める。
// 親だけを止めると、孫が標準出力をつかんだまま残り、出力の読み取りが終わらない。
func killGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
}
