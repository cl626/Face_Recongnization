# function reading
1. evaluate: 主函数，其中calculate_roc计算ROC曲线，calculate_val计算值和标准差
2. calculate_roc：distance 计算待比较嵌入表示的距离，calculate_accuracy根据门限值计算准确度，is_fp,is_fn，选择准确度最高的一折，画出对应的ROC
3. calculate_val：线性拟合出门限，之后calcuulate_far按小于threshold
* question: 算val,far分母运气不好为0-->加1平滑
4. 效果很烂，准确率极低，怎么把准确率调上去?
5. ~~训练的准确度都只有55.6%--~~    已解决读pairs.txt时正负判反了
6. 对K-折交叉验证不熟悉，这里每一折再从n个门限里选择最好的
~~# 自己搭一个简单的跑结果~~