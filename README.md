1. 解压NewProject/datas/* 目录下的文件
2. 执行train.py开始训练模型(*)
3. 到result中检查下列文件是否成功保存了
   - History3d.json: 训练过程中的loss的数据
   - model3d.h5: 神经网络参数
   - dice.png: 每帧的复原精度(DSC)的时间推移
4. 执行test.py测试

(*): 执行之前需要用anaconda配置tensorflow-gup环境