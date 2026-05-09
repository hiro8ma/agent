// ベクトル = 数値の 1 次元配列
// 例: [1, 0]    （2 次元ベクトル）
// 例: [1, 2, 3] （3 次元ベクトル）
export type Vector = number[];

// 行列 = ベクトルの配列（2 次元配列）
// 例: [[1, 0], [0, 1]]   （2x2 行列、2 行 2 列）
//
// Attention で扱う行列の形
//   Q: [n_q, d_k]   n_q 個の Query、各 Query は d_k 次元
//   K: [n_k, d_k]   n_k 個の Key、各 Key は d_k 次元
//   V: [n_k, d_v]   n_k 個の Value、各 Value は d_v 次元
export type Matrix = number[][];
