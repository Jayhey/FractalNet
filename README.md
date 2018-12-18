# Fractalnet in tensorflow


모델 학습 명령어는 다음과 같습니다. 데이터는 cifar10을 이용하며 augmentation은 따로 적용하지 않고  학습을 진행합니다.

```
python FractalNet.py
```

첨부된 파일은 다음과 같습니다.
- FractalNet.py : 실행 파일
- fractal_module.py : FractalNet.py 실행에 필요한 모듈


#### Default option 
- Epochs : 200
- Batch size : 100
- Number of blocks : 5 (max 5)
- Number of columns : 4 

위 옵션은 사용자가 명령행 옵션을 사용하여 임의로 변경할 수 있습니다. 
```
python FractalNet.py --help
```
모든 옵션은 논문과 동일하게 설정하였습니다. 

- Conv filters in each block : 64, 128, 256, 512, 512
- Dropout rate in each block : 0, 0.1, 0.2, 0.3, 0.4
- Local dropout rate : 0.15
- Optimizer : SGD with momentum 0.9


#### Loss & Accuracy
Loss,accuray 그래프 구조 모두 텐서보드에서 확인할 수 있습니다. 

```
tensorboard --logdir="./"
```

#### Fractal block
네트워크를 구성하는 fractal block 구조는 아래 그림과 같습니다. 
![](/imgs/fractal_block.png)

