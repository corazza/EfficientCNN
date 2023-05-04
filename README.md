# EfficientCNN

ResNet-18: Tucker decomposition applied to every Conv2D weight tensor.

## Results

- model size: 12.1788% of original
- training time: 279.5585s vs. 487.7116s
- accuracy: 77.01% vs. 83.15%

```txt
Tuckerification...
No. parameters original = 11689512
No. parameters tuckerified = 1423647 (12.1788%)
Training original model for 20 epochs
Epoch 1, Loss: 1.2216
...
Epoch 20, Loss: 0.3519
Done, accuracy = 83.1500%, training time = 487.7116s
Training tuckerified model for 20 epochs
Epoch 1, Loss: 1.6815
...
Epoch 20, Loss: 0.6179
Done, accuracy = 77.0100%, training time = 279.5585s
```
