# MateGen：交互式智能编程助手

![136439902d507ef41f9f746bddd47fc](https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/136439902d507ef41f9f746bddd47fc.jpg)

<p align="center">
  <a href="README.md"><strong>English</strong></a> | 
  <a href="docs/MateGen使用教程.ipynb"><strong>MateGen使用教程</strong></a> | 
  <a href="docs/wechat.png"><strong>微信</strong></a>
</p>

## MateGen是什么？

​	**MateGen是由九天老师大模型教研团队开发的交互式智能编程助手（Interactive Intelligent Programming Assistant），可在Jupyter代码环境中便捷调用，辅助用户高效完成智能数据分析、机器学习、深度学习和大模型开发等工作，并可根据用户实际需求定制知识库和拓展功能。**

MateGen基础功能如下：

- 🤖**高易用性，零门槛调用**：MateGen为用户提供在线大模型应用服务，**无需任何硬件或网络代理门槛**，即可一键安装并开启对话；
- 🚀**强悍的高精度RAG系统**：一键同步本地文档并进行RAG检索问答，**最多支持1000篇文档以及10G文档内容进行检索**，支持md、ppt、word、pdf等主流文档格式高精度问答，能够高效率实现包括海量文档总结、大海捞针内容测试、情感倾向测试问答等功能。MateGen可根据用户问题自动识别是否需要进行RAG检索；
- 🏅**本地Python代码解释器**：可连接用户本地Python环境完成编程任务，包括数据清洗、数据可视化、机器学习、深度学习、大模型开发等代码工作编写，支持先学习代码库再进行编程、能够根据实际情况debug，支持可视化图片自动上传图床等功能；
- 🚩**高精度NL2SQL功能**：可根据用户需求编写SQL，并连接本地MySQL环境自动执行，可自动debug，并且支持先检索数据字典、企业数据知识库再进行SQL编写，从而提高SQL编写精度；
- 🛩️**视觉能力和联网能力**：对话时输入图片网址即可开启MateGen视觉能力对图片内容进行识别，同时MateGen也具备联网能力，当遇到无法回答的问题时，可自动开启搜索问答模式；
- 🚅**无限对话上下文**：MateGen拥有无限上下文对话长度，MateGen会根据历史对话的未知信息密度进行合理处理，从而在节省token的同时实现无限对话上线文。
- 💰**极低的使用成本**：尽管MateGen由在线大模型驱动，但实际使用成本极低，普通模式下50万token仅需1元！

- 除此之外，MateGen具备**高稳定性**与**高可用性**，同时支持**Multi Function calling**（一个任务开启多个功能）和**Parallel Function calling**（一个功能开多个执行器），能够**自动分解复杂任务**、**自动Debug**，并且拥有一定程度“**自主意识**”，能够**审查自身行为**并深度**挖掘用户意图**。

## MateGen API-KEY获取

​	MateGen目前只上线了在线服务版本，借助在线大模型来完成各项服务，无需本地硬件、无需网络环境要求即可零门槛使用。**调用MateGen需要通过API-KEY进行身份验证**，测试阶段限量**免费开放3亿免费token额度，送完即止，API-KEY领取、加入技术交流群、其他任何问题，<span style="color:red;">扫码添加客服小可爱(微信：littlelion_1215)，回复“MG”详询哦👇</span>**

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713010710534.png" alt="image-20240713010710534" width="200"/>
</div>

**欢迎课程学员和新老用户多多支持本项目，项目star超过10k即上线开源版及教学教程！**

## MateGen使用效果演示

注：各项演示操作可参考[《MateGen使用教程》](docs/MateGen使用教程.ipynb)中相关代码来实现。

- **零门槛便捷调用**

​	只需三步即可在Jupyter中调用MateGen：**安装、导入、对话**！

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712185454166.png" alt="image-20240713010710534" width="800"/>
  </div>

- **本地海量文本知识库问答**

  ​	借助MateGen，可实现高精度本地知识库问答，MateGen在RAG系统最多**支持1000个文本+10G规模**文本检索！

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712212936156.png" alt="image-20240713010710534" width="600"/>
  </div>
  
- **交互式可视化绘图**

  ​	MateGen同时具备视觉能力和本地代码解释器功能，因此可以**根据用户输入的图片，模仿绘制**！

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712193604820.png" alt="image-20240713010710534" width="600"/>
  </div>

- **高精度NL2SQL**

  ​	MateGen支持**全自动RAG+NL2SQL联合执行**，因此可以**先从知识库中了解数据集字段信息和业务信息然后再编写SQL，并且支持自动审查与自动debug**，从而大幅提高SQL准确率。

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712222633885.png" alt="image-20240713010710534" width="600"/>
  </div>

- **自动机器学习**

  ​	MateGen支持**全自动RAG+代码解释器**联合执行，支持先阅读企业机器学习代码库再进行机器学习建模，通过**自然语言一键调用不同机器学习建模策略**，创建你的机器学习“贾维斯”。

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712225738564.png" alt="image-20240713010710534" width="600"/>
  </div>  
  
- **前沿深度学习论文解读与架构复现**

  ​	基于自身强大的RAG系统以及Multi-Function功能，MateGen能够进行深度**论文辅导**，可以**帮助用户逐段翻译和解读论文—>总结论文核心知识点—>一键编写百行代码代码复现论文架构并在本地代码环境直接运行验证**！

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712231230708.png" alt="image-20240713010710534" width="600"/>
  </div>

- **Kaggle竞赛辅导**

  ​	借助MateGen的联网能力+知识库问答能力+NL2Python能力，MateGen还可**辅助用户参与Kaggle竞赛**。MateGen可以根据用户提供的赛题，**自动获取赛题解释与数据集解释信息，自动爬取赛题高分Kernel并组建竞赛知识库，然后辅助用户进行竞赛编程，并自动提交比赛结果至Kaggle平台，最终根据提交结果提示用户调整竞赛策略，从而冲击更高分数！**

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713003522041.png" alt="image-20240713010710534" width="600"/>
  </div>  

- **智能助教**

  ​基于MateGen强悍的海量文本检索问答能力以及代码能力，一个储备了课程课件知识库的MateGen完全可以作为一名智能助教，学习前可以辅助用户进行课前预习、指定学习计划，学习中可以7*24小时实时答疑、随时辅助用户完成编程或其他代码任务，课后还可以根据用户课中提问来编写习题、分析用户薄弱知识点并将其总结为课后复习文档！
  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240726003714902.png" alt="image-20240713010710534" width="600"/>
  </div>  

MateGen更多应用场景即将上线。

## MateGen安装部署流程

​	MateGen项目轻便、调用简单，可使用pip直接进行安装，同时调用风格类似于sklearn，实例化一个MateGen智能体即可直接开启对话！

- **MateGen下载方法**

​	MateGen现已在PyPI平台上线，可以直接通过`pip install mategen`进行安装，需要注意的是，MateGen运行所需依赖较多，因此推荐借助虚拟环境进行安装。首先创建一个名为`mategen`的虚拟环境：        

```bash
conda create -n mategen python=3.8
```

然后使用如下指令激活虚拟环境：

```bash
conda activate mategen
```

接着在虚拟环境中安装MateGen：

```bash
pip install mategen
```

安装完成之后，考虑到需要在Jupyter中调用MateGen，我们还需要在虚拟环境中安装IPython Kernel：

```bash
pip install ipykernel
```

并且将这个虚拟环境添加到Jupyter的Kernel列表中：

```bash
python -m ipykernel install --user --name mategen --display-name "mategen"
```

然后开启Jupyter服务：     

```bash
jupyter lab
```

然后在Jupyter的Kernel中选择mategen，即可进入到对应虚拟环境运行MateGen：

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713012151873.png" alt="image-20240713010710534" width="400"/>
  </div>  

- **MateGen调用方法**

  ​	MateGen调用过程非常简单，只需要在代码环境中导入并输入有效的API-KEY即可开启对话！

  ```python
  mategen = MateGenClass(api_key = 'YOUR_API_KEY')
  ```

  然后即可使用chat功能进行单次对话或多轮对话：

  ```python
  mategen.chat("你好，很高兴见到你！")
  ```

  ```markdown
  ▌ MateGen初始化完成，欢迎使用！
  
  你好！很高兴见到你！有什么我可以帮助你的吗？
  ```

  更多MateGen使用方法，详见[《MateGen使用教程》](docs/MateGen使用教程.ipynb)。

  ​	免费API获取👉MateGen目前正处于测试阶段，限量**免费开放3亿免费token额度，送完即止，API-KEY领取、加入技术交流群、其他任何问题，<span style="color:red;">扫码添加客服小可爱(微信：littlelion_1215)，回复“MG”详询哦👇</span>**

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713010710534.png" alt="image-20240713010710534" width="200"/>
  </div>

## MateGen架构与应用说明

- MateGen基本架构

  ​	MateGen采用了目前最先进的threads-runs架构，以更好的进行用户历史消息对话管理以及自动修复运行中遇到的各种问题，同时采用了client与server分离架构，以确保最大程度Agent运行稳定性，同时支持多种不同类型底层大模型，MateGen基本结构如下：

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240715001340035.png" alt="image-20240713010710534" width="600"/>
</div>

- MateGen用于智能助教

  ​	MateGen同时可适用于多种不同类型具体业务场景，例如MateGen现已用于九天老师团队各门课程辅助教学环节，用于智能助教。MateGen充当智能助教的基本功能执行流程如下：

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240715001720425.png" alt="image-20240713010710534" width="700"/>
</div>
  