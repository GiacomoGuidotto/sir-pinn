## TODO

- Find an activation function that constraints the output to be inside a range:
  e.g: beta needs to be positive and with an upper bound around 10 * 0.2 (r0 *
  delta)
- Create a twin model with variable constraints
- Rescale population correctly, and using and activation, not a scaling constant

patience = | dataset | / batch_size
data * 0 = 0