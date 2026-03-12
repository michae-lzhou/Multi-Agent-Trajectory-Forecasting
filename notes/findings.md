
# Findings 1
## Time: 2026-03-04
## Method: LSTM, LSTM_NCap, LSTM+Attention

### LSTM:
#### Best Recipe: last_predict, hidden = 256, layers = 2, converge at 38 epoch
teacher-forcing performs bad because we feed ground_truth on training but last_predict in testing
hidden dims, layers don't affect the performance too much.

### LSTM_NCap
current LSTM structure doesn't use the information of neighbors


### LSTM+Attention
