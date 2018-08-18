KDD18 London: Deep Learning and Natural Language Processing with Apache MXNet Gluon
===================================================================================

<h3>Time: Tuesday, August 21, 2018<br/>Location: ICC Capital Suite Room 14+15+16</h3>

<span style="color:grey">Presenter: Alex Smola, Leonard Lausen, Haibin Lin</span><br/>

Abstract
--------

While deep learning has rapidly emerged as the dominant approach to training predictive models for large-scale machine learning problems, these algorithms push the limits of available hardware, requiring specialized frameworks optimized for GPUs and distributed cloud-based training. Moreover, especially in natural language processing (NLP), models contain a variety of moving parts: character-based encoders, pre-trained word embeddings, long-short term memory (LSTM) cells, and beam search for decoding sequential outputs, among others.

This tutorial introduces GluonNLP, a powerful new toolkit that combines MXNet's speed, the user-friendly Gluon frontend, and an extensive new library automating the most painful aspects of deep learning for NLP. In this full-day tutorial, we will start off with a crash course on deep learning with Gluon, covering data, autodiff, and deep (convolutional and recurrent) neural networks. Then we'll dive into GluonNLP, demonstrating how to work with word embeddings (both pre-trained and from scratch), language models, and the popular Transformer model for machine translation.


Preparation
-----------
While most of the notebooks we prepared can run from the comfort of your laptop, some more interesting real-life problems benefit from more computation power. At the tutorial, we will provide each participant with a $50 AWS credit code for you to try out these problems yourself on the Amazon EC2 machines.

In preparation for the hands-on tutorial, **please make sure that you have an AWS account with at least one p2.8xlarge AND one p3.2xlarge instance in EU (Ireland) available for launch**. You may register an AWS account and **follow the instructions on [this page](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) to verify and request for p2.8xlarge AND one p3.2xlarge instance limits** in **EU (Ireland)** region.


Agenda
------
| Time        | Title                                                         |
|-------------|---------------------------------------------------------------|
| 8:30-9:15   | Installation and Basics (NDArray, AutoGrad, Libraries)        |
| 9:15-9:30   | Neural Networks 101 (MLP, ConvNet, LSTM, Loss, SGD) - Part I  |
| 9:30-10:00  | Coffee Break                                                  |
| 10:00-10:30 | Neural Networks 101 (MLP, ConvNet, LSTM, Loss, SGD) - Part II |
| 10:30-11:00 | Computer Vision 101 (GluonCV)                                 |
| 11:00-11:30 | Parallel and Distributed Training                             |
| 11:30-12:00 | Data I/O in NLP (and Iterators)                               |
| 12:00-13:30 | Lunch Break                                                   |
| 13:30-14:15 | Embeddings                                                    |
| 14:15-15:00 | Language Models (LM)                                          |
| 15:00-15:30 | Sequence Generation from LM                                   |
| 15:30-16:00 | Coffee Break                                                  |
| 16:00-16:15 | Sentiment Analysis                                            |
| 16:15-17:00 | Transformer Models and Machine Translation                    |
| 17:00-17:30 | Questions                                                     |

Have questions? Contact us at [amazonai-kdd18@amazon.com](mailto:amazonai-kdd18@amazon.com)
