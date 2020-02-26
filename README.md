# 问题背景：
在一片区域内，有一个移动的robot和几个固定的landmark. robot与几个landmark之间的距离可以实时计算获取。
在这种条件下，通过robot与landmarks之间的距离变化，对该robot的具体位置进行定位。
具体定位方法是partile filtering.

## 要求：
请理解代码，并尝试如下改造：
1). 基于particles的位置，计算最终定位robot的唯一位置并输出到屏幕上，说明计算方式
2). 修改weights的分布为帕累托分布（当前使用的是正态分布）
3). 为landmark和robot之间的距离增加随机误差，观察定位结果
*4). 修改particle filtering过程，消除随机误差对定位结果的影响（不一定完成，可讨论思路）