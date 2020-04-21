
一、语料库中的错误 <br>
1.时唱接z5歌 ----> 时唱接籬歌。 <br>
2.屈rH猛--------->  屈赟猛，    <br>
3.花门kO面请雪耻-------> 花门剺面请雪耻  <br>
4.权门多噂eR---------->权门多噂沓  <br>
5.苞枿ba矣----------->苞枿薱矣   <br>
6.翠微zc叶垂鬓唇  -------->翠微盍叶垂鬓唇 <br>

二、实验结果

1. 经过多次实验，发现LSTM中的hidden_size设为512，embedding_size设为512是比较好的选择
  （hidden_size过小时，生成的句子不能很好的断句，会出现长句子，即没有空格的情况，可以参考hidden_size设为256的几次实验）

2. Epoch没有必要设的太大。实验中发现在大约第70个Epoch时，困惑度（perplexity）就可以降到1.5以下，而困惑度小于2.0就可以开始生成新句子<br>
  （参考：https://github.com/L1aoXingyu/Char-RNN-PyTorch）

3. 生成结果中会出现固定句子或短语重复的现象， 比如：<br>

    萧萧山路穷秋雨 人醉寄教仙去 罗军青丝五月 骅骝作已散 银鞍却覆香罗帕来 银鞍却覆香罗帕来 义公习已散 银鞍却覆香罗帕来 义公已香开 五陵佳气郁葱葱萋萋长       河庐春水生 雨     泻寒月 喜逢金马客 天水相与散 池君凤初来 今日龙门下 馆汗流血促弟开 银鞍却覆香罗帕来 去水相与永 怀新目似击 接要面月 天水相与永 怀     新目似击 接要面月 天水相与永 怀新     目似击 接要面月 天子命元帅 奋其雄图  尘遮晚日红 拂水低徊舞罗散龙<br>
   （来自：epoch_200_dropout_0.5_hidden_size_1024_embedding_size_512_batch_size_64_len_seq_20_lr_0.006 2_layers）

   可以看见：<br>
  （1）“银鞍却覆香罗帕来”出现多次，而且不是连续出现。这句话也在下面一个毫无相关的生成段落中出现了，但在语料库中，“银、鞍”各自只出现了5、6次  <br>
  （原始语料为“银鞍却覆香罗帕”，后接一个“来”，猜测是因为“来”出现的频率比较高，有71次）<br>
  
   （2）“天水相与永 怀新目似击 接要面月” 出现多次，且是连续出现，猜测是overfitting了 <br>
   （原始语料为“天水相与永。怀新目似击”，不是一句诗。后面接“接要面月”，不知是什么意思）
