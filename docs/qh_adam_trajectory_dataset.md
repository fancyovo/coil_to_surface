# QH Adam 合成轨迹数据集

数据集的一条数据对应一次完整的搜索：先按 QUASR QH 训练集的
`(nfp, n_base_coils)` 联合经验分布抽一个条件，从标准高斯潜空间抽
32 个起点，用无历史的全局磁轴搜索评分后选取最高的有效起点，再运行
200 步 Adam。Adam 固定为当前已验证的性价比配置：FP32 RK4-128、
64 个新的正交随机方向、128 个中心差分端点、$h=0.005$、
$\eta=0.02$、$\beta_1=0.7$、$\beta_2=0.999$。

每条数据目录中，`screening/` 保存32个起点的潜变量、解码线圈和完整原生
score；`optimization/training_trace.npz` 保存每步的中心潜变量与线圈、
64 个方向、128 个解码端点、端点 score 及分量、估计梯度、Adam 一阶/
二阶矩、建议更新和实际更新。端点潜变量可从中心、方向和 $h$
精确重建。`center_native_results.jsonl.gz` 保存初始中心与后续所有中心的
完整原生评分。

数据集为只追加结构。每张 GPU 独立生产完整轨迹，在
`incomplete/<id>.partial/` 内完成全部计算和自检后，才原子移入
`trajectories/<id>/`。未满200步或全无有效起点的尝试进入 `failures/`，
不参与训练数据统计。各 GPU 流的进度位于 `streams/*/progress.json`。
同一程序错误连续三次时该流会停止，防止 ABI 或路径错误长时间空转。
