package model

type Order struct {
	ID            string `json:"id"`
	CustomerName  string `json:"customerName"`
	PaymentMethod string `json:"paymentMethod"`
	AmountJPY     int    `json:"amountJpy"`
}
