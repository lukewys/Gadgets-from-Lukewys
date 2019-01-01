# 风格识别（风格距离测量）-放弃
对于符号音乐（即谱子，MIDI）风格距离测量的尝试，也可以进行风格识别。

初步设想对音乐序列加窗，并使用时间序列相关分析，但似乎无效。
对于现有的数据集，其自相关函数的平均值比互相关函数的要小。

具体思考见笔记.docx。

# Genre classify（Genre distance measurement）-deprecated
Attempting to measure the genre distance in terms of symbolic music (score，MIDI), 
could also use to classify music genre.

The initial thinking is to apply window function to melody sequence,
 and calculate the correlation. However, it seems this method is invalid. 

As for the dataset I got, the correlation between the pieces in different genre is 
higher than the correlation between the pieces in same genre.

Detailed thinking, see: 笔记.docx。