# MateGen: Next-Generation Interactive Intelligent Programming Assistant

![136439902d507ef41f9f746bddd47fc](https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/136439902d507ef41f9f746bddd47fc.jpg)

<p align="center">
  <a href="README_zh.md"><strong>简体中文</strong></a> | 
  <a href="docs/MateGen使用教程.ipynb"><strong>MateGen使用教程</strong></a> | 
  <a href="docs/wechat.png"><strong>微信</strong></a>
</p>

## What is MateGen?

**MateGen is an Interactive Intelligent Programming Assistant developed by the Jiutian Teacher's Large Model Research Team. It can be conveniently used within the Jupyter coding environment to assist users in efficiently completing tasks related to intelligent data analysis, machine learning, deep learning, and large model development. MateGen also offers customizable knowledge bases and extendable functionalities to meet the specific needs of users.**

The basic features of MateGen are as follows:

- 🤖 **High Usability, Zero Threshold**: MateGen provides online large model application services for users, enabling one-click installation and initiation of interactions without any hardware or network proxy barriers.
- 🚀 **Robust High-Precision RAG System**: Synchronize local documents with one click and conduct RAG (Retrieval-Augmented Generation) searches and Q&A. MateGen supports the retrieval of up to 1000 documents and 10GB of document content, providing high-precision Q&A for mainstream document formats such as md, ppt, word, and pdf. It efficiently handles functions like massive document summarization, needle-in-a-haystack content testing, and sentiment analysis Q&A. MateGen can automatically determine whether RAG search is required based on user queries.
- 🏅 **Local Python Code Interpreter**: Connect to the user's local Python environment to complete programming tasks, including data cleaning, data visualization, machine learning, deep learning, and large model development. It supports learning the code repository before programming, debugging based on actual scenarios, and automatically uploading visualized images to image hosting services.
- 🚩 **High-Precision NL2SQL Functionality**: Generate SQL queries based on user needs and connect to the local MySQL environment for automatic execution and debugging. It also supports retrieving data dictionaries and enterprise data knowledge bases before writing SQL queries, thereby improving SQL accuracy.
- 🛩️ **Visual and Networking Capabilities**: Enable MateGen's visual capabilities by inputting image URLs during conversations to recognize image content. MateGen also has networking capabilities to automatically initiate search-based Q&A when encountering questions it cannot answer.
- 🚅 **Unlimited Conversation Context**: MateGen supports unlimited conversation length, managing the density of unknown information in historical dialogues to save tokens while maintaining an infinite conversation context.
- 💰**Extremely Low Usage Cost**: Although MateGen is powered by an online large model, the actual usage cost is extremely low. In normal mode, 500,000 tokens cost only 1 yuan!

Additionally, MateGen boasts **high stability** and **high availability**, supports **Multi Function calling** (one task triggering multiple functions) and **Parallel Function calling** (one function initiating multiple executors), can **automatically decompose complex tasks**, **auto-debug**, and possesses a certain degree of **autonomous awareness**. It can **review its own actions** and deeply **explore user intentions**.

## MateGen API-KEY Acquisition

MateGen is currently available only as an online service, utilizing large models to deliver various services without the need for local hardware or network environment requirements, enabling zero-threshold usage. **To use MateGen, an API-KEY is required for authentication**. During the testing phase, a limited **3 billion free token quota** is available, distributed on a first-come, first-served basis. For API-KEY acquisition, joining the technical support group, or any other inquiries, <span style="color:red;">please scan the QR code to add our friendly customer service representative on WeChat (ID: littlelion_1215) and reply with "MG" for more details👇</span>.

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713010710534.png" alt="image-20240713010710534" width="200"/>
</div>

**We welcome and appreciate the support of course participants and new and old users alike. If the project receives more than 10k stars, we will release an open-source version along with instructional tutorials!**

## MateGen Usage Demonstration

Note: Refer to the [MateGen Usage Tutorial](docs/MateGen使用教程.ipynb) for the relevant code for each demonstration.

- **Zero-Threshold Convenient Invocation**

  ​	Invoke MateGen in Jupyter with just three steps: **Install, Import, and Interact**!
  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712185454166.png" alt="image-20240713010710534" width="800"/>
  </div>
- **Local Massive Text Knowledge Base Q&A**

  ​        With MateGen, achieve high-precision local knowledge base Q&A. MateGen's RAG system supports the retrieval of **up to 1000 texts and 10GB of text content**!
  
  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712212936156.png" alt="image-20240713010710534" width="600"/>
  </div>
  
- **Interactive Visualization Drawing**

  ​        MateGen also possesses visual capabilities and a local code interpreter function, enabling it to **mimic and draw based on user-input images**!

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712193604820.png" alt="image-20240713010710534" width="600"/>
  </div>
  
- **High-Precision NL2SQL**

  ​        MateGen supports **fully automated RAG+NL2SQL joint execution**, allowing it to **first understand dataset fields and business information from the knowledge base before writing SQL, and supports automatic review and debugging**, significantly improving SQL accuracy.

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712222633885.png" alt="image-20240713010710534" width="600"/>
  </div>
  
- **Automated Machine Learning**

  ​        MateGen supports **fully automated RAG+code interpreter** joint execution, enabling it to read the enterprise machine learning code repository before modeling. It allows **one-click invocation of different machine learning modeling strategies through natural language**, creating your machine learning "Jarvis".

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712225738564.png" alt="image-20240713010710534" width="600"/>
  </div>  

- **Advanced Deep Learning Paper Interpretation and Architecture Reproduction**

  ​       Leveraging its powerful RAG system and Multi-Function capabilities, MateGen can conduct in-depth **paper tutoring**. It can **assist users in translating and interpreting papers section by section, summarizing core knowledge points, writing hundreds of lines of code to reproduce paper architectures with one click, and directly running and verifying them in the local code environment**!

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240712231230708.png" alt="image-20240713010710534" width="600"/>
  </div>

- **Kaggle Competition Coaching**

  ​        Utilizing MateGen's networking capability, knowledge base Q&A ability, and NL2Python capability, MateGen can **assist users in participating in Kaggle competitions**. MateGen can automatically **retrieve competition explanations and dataset information based on the user's provided competition problems, scrape high-scoring kernels for the competition, build a competition knowledge base, assist users in competition programming, and automatically submit results to the Kaggle platform. It will then suggest adjustments to the competition strategy based on the submission results, aiming for higher scores!**

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713003522041.png" alt="image-20240713010710534" width="600"/>
  </div>  

- **Intelligent Teaching Assistant**

Based on MateGen's powerful capabilities in massive text retrieval and Q&A, as well as its coding abilities, a MateGen equipped with a knowledge base of course materials can fully function as an intelligent teaching assistant. Before learning, it can assist users with pre-study preparations and create study plans. During learning, it can provide 24/7 real-time Q&A support, helping users with programming or other coding tasks anytime. After class, it can generate exercises based on users' questions during the course, analyze weak points in their knowledge, and compile them into review documents for post-class revision.

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240726003714902.png" alt="image-20240713010710534" width="600"/>
  </div>

More application scenarios for MateGen are coming soon.

## MateGen Installation and Deployment Process

​	MateGen is lightweight and easy to call. It can be installed directly using pip, and its invocation style is similar to sklearn. By instantiating a MateGen agent, you can start interacting right away!

- **MateGen Download Method**

  MateGen is now available on the PyPI platform and can be installed directly via `pip install mategen`. Note that MateGen requires many dependencies to run, so it is recommended to use a virtual environment for installation. First, create a virtual environment named `mategen`:

  ```python
  conda create -n mategen python=3.8
  ```

  Then activate the virtual environment with the following command:

  ```python
  conda activate mategen
  ```

  Next, install MateGen in the virtual environment:
  
  ```python
  pip install mategen
  ```
  
  After installation, considering the need to invoke MateGen in Jupyter, we need to install the IPython Kernel in the virtual environment:
  
  ```python
  pip install ipykernel
  ```
  
  Add this virtual environment to Jupyter's Kernel list:
  
  ```python
  python -m ipykernel install --user --name mategen --display-name "mategen"
  ```
  
  Then start the Jupyter service:
  
  ```python
  jupyter lab
  ```
  
  Select the mategen kernel in Jupyter to enter the corresponding virtual environment and run MateGen:

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713012151873.png" alt="image-20240713010710534" width="400"/>
  </div>  

- **MateGen Invocation Method**

  Invoking MateGen is very simple. Just import it in the code environment and input a valid API-KEY to start interacting!

  ```python
  mategen = MateGenClass(api_key = 'YOUR_API_KEY')
  ```

  Then you can use the chat function for single or multi-turn conversations:

  ```python
  mategen.chat("你好，很高兴见到你！")
  ```

  ```markdown
  ▌ MateGen初始化完成，欢迎使用！
  
  你好！很高兴见到你！有什么我可以帮助你的吗？
  ```

  For more usage methods of MateGen, refer to the [MateGen Usage Tutorial](docs/MateGen使用教程.ipynb).

  ​        Free API acquisition 👉 MateGen is currently in the testing phase, with a limited **free quota of 3 billion tokens, available while supplies last. For API-KEY acquisition, joining the technical support group, or any other inquiries, <span style="color:red;">please scan the QR code to add our friendly customer service representative on WeChat (ID: littlelion_1215) and reply with "MG" for more details👇</span>**

  <div align="center">
  <img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713010710534.png" alt="image-20240713010710534" width="200"/>
  </div>

## MateGen Architecture and Application Description

- Basic Architecture of MateGen

​        MateGen adopts the most advanced threads-runs architecture for better management of user historical message conversations and automatic resolution of various issues encountered during operation. Additionally, it utilizes a client-server separation architecture to ensure maximum stability of the Agent operation while supporting various types of underlying large models. The basic structure of MateGen is as follows:

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240715001340035.png" alt="image-20240713010710534" width="600"/>
</div>

- MateGen for Intelligent Teaching Assistant

​        MateGen can be applied to various specific business scenarios. For example, it is currently used in the Jiutian Teacher's team to assist in the teaching of various courses, serving as an intelligent teaching assistant. The basic functional execution process of MateGen as an intelligent teaching assistant is as follows:

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240715001720425.png" alt="image-20240713010710534" width="700"/>
</div>