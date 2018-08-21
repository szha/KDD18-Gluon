KDD18 London: Deep Learning and Natural Language Processing with Apache MXNet Gluon
===================================================================================

<h3>Time: Tuesday, August 21, 2018<br/>Location: ICC Capital Suite Room 14+15+16</h3>

<span style="color:grey">Presenter: Alex Smola, Leonard Lausen, Haibin Lin</span><br/>

Abstract
--------

While deep learning has rapidly emerged as the dominant approach to training predictive models for large-scale machine learning problems, these algorithms push the limits of available hardware, requiring specialized frameworks optimized for GPUs and distributed cloud-based training. Moreover, especially in natural language processing (NLP), models contain a variety of moving parts: character-based encoders, pre-trained word embeddings, long-short term memory (LSTM) cells, and beam search for decoding sequential outputs, among others.

This tutorial introduces [GluonNLP](http://gluon-nlp.mxnet.io/) ([GitHub](https://github.com/dmlc/gluon-nlp)), a powerful new toolkit that combines MXNet's speed, the user-friendly Gluon frontend, and an extensive new library automating the most painful aspects of deep learning for NLP. In this full-day tutorial, we will start off with a crash course on deep learning with Gluon, covering data, autodiff, and deep (convolutional and recurrent) neural networks. Then we'll dive into [GluonNLP](http://gluon-nlp.mxnet.io/), demonstrating how to work with word embeddings (both pre-trained and from scratch), language models, and the popular Transformer model for machine translation.


Preparation
-----------
While most of the notebooks we prepared can run from the comfort of your laptop, some more interesting real-life problems benefit from more computation power. At the tutorial, we will provide each participant with a $50 AWS credit code for you to try out these problems yourself on the Amazon EC2 machines.

In preparation for the hands-on tutorial, **please make sure that you have an AWS account with at least one p2.8xlarge AND one p3.2xlarge instance in EU (Ireland) available for launch**. You may register an AWS account and **follow the instructions on [this page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) to verify and request for p2.8xlarge AND one p3.2xlarge instance limits** in **EU (Ireland)** region.


Agenda
------
| Time        | Title                                                         | Slides    | Notebooks  |
|-------------|---------------------------------------------------------------|-----------|------------|
| 8:30-9:15   | Installation and Basics (NDArray, AutoGrad, Libraries)        | [link][1] | [link][00] [link][01] |
| 9:15-9:30   | Neural Networks 101 (MLP, ConvNet, LSTM, Loss, SGD) - Part I  | [link][2] | [link][02] |
| 9:30-10:00  | Coffee Break                                                  |           |            |
| 10:00-10:30 | Neural Networks 101 (MLP, ConvNet, LSTM, Loss, SGD) - Part II |           | [link][02] |
| 10:30-11:00 | Computer Vision 101 (GluonCV)                                 | [link][3] | [link][03] |
| 11:00-11:30 | Parallel and Distributed Training                             | [link][4] | [link][04] |
| 11:30-12:00 | Data I/O in NLP (and Iterators)                               |           | [link][05] |
| 12:00-13:30 | Lunch Break                                                   |           |            |
| 13:30-14:15 | Embeddings                                                    | [link][5] | [link][06] |
| 14:15-15:00 | Language Models (LM)                                          | [link][5] | [link][07] |
| 15:00-15:30 | Sequence Generation from LM                                   | [link][6] | [link][08] |
| 15:30-16:00 | Coffee Break                                                  |           |            |
| 16:00-16:15 | Sentiment Analysis                                            |           | [link][09] |
| 16:15-17:00 | Transformer Models and Machine Translation                    | [link][7] | [link][10] |
| 17:00-17:30 | Questions                                                     |           |            |

Have questions? Contact us at [amazonai-kdd18@amazon.com](mailto:amazonai-kdd18@amazon.com)

[1]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%201.pdf
[2]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%202.pdf
[3]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%203.pdf
[4]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%204.pdf
[5]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%205.pdf
[6]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%206.pdf
[7]: https://github.com/szha/KDD18-Gluon/blob/master/slides/KDD%20Tutorial%207.pdf
[00]: https://github.com/szha/KDD18-Gluon/tree/master/00_setup
[01]: https://github.com/szha/KDD18-Gluon/tree/master/01_basics
[02]: https://github.com/szha/KDD18-Gluon/tree/master/02_neural_networks
[03]: https://github.com/szha/KDD18-Gluon/tree/master/03_computer_vision
[04]: https://github.com/szha/KDD18-Gluon/tree/master/04_distributed_training
[05]: https://github.com/szha/KDD18-Gluon/tree/master/05_data_pipeline
[06]: https://github.com/szha/KDD18-Gluon/tree/master/06_word_embedding
[07]: https://github.com/szha/KDD18-Gluon/tree/master/07_language_model
[08]: https://github.com/szha/KDD18-Gluon/tree/master/08_sequence_generation
[09]: https://github.com/szha/KDD18-Gluon/tree/master/09_sentiment_analysis
[10]: https://github.com/szha/KDD18-Gluon/tree/master/10_translation
