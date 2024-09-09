"""Experiment-running framework."""
import argparse
import importlib
from logging import debug

import numpy as np
from pytorch_lightning.trainer import training_tricks
import torch
import pytorch_lightning as pl
import lit_models
import yaml
import time
from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel
from pytorch_lightning.plugins import DDPPlugin
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁止分词器tokenizer的并行处理，避免死锁等问题的出现


# In order to ensure reproducible experiments, we must set random seeds.

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'可以在运行时根据字符串来导入和使用类，而不需要在代码中显式地导入它们"""
    module_name, class_name = module_and_class_name.rsplit(".", 1) # 'data.WIKI80'
    module = importlib.import_module(module_name)  # 使用 importlib 模块的 import_module 函数动态地导入名为 module_name 的模块, module: <module 'data' from '/root/KnowPrompt/./data/__init__.py'>
    class_ = getattr(module, class_name)  # 从导入的模块 module 中获取名为 class_name 的类, <class 'data.dialogue.WIKI80'>
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)  # 创建命令行参数解析器

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)  # 将pl.Trainer的参数添加进parser
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access,将trainer_parser的第二组参数的参数组名称改为Trainer Args
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser]) #  创建新解析器parser，这个解析器继承了trainer_parser的参数

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)  # action="store_true"的含义是，当命令行包含了特定选项时，参数的值将被设置为 True，否则参数的值将保持默认值，这里默认值通过default=False被设置为了false
    parser.add_argument("--litmodel_class", type=str, default="TransformerLitModel")   # 这个参数用于指定脚本中使用的PyTorch Lightning LitModel模型（Lightning 模型），这个litmodel并不是我们平时所说的“模型”，我们平时所说的“模型”在11中指定，在11中我们也会介绍model和litmodel的区别
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="DIALOGUE")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="bert.BertForSequenceClassification") # 指定训练模型使用的模型，默认使用RobertaForPrompt这个模型
    # model定义了神经网络的架构（例如，层数、隐藏单元数、注意力机制等）和它们如何处理输入数据（即前向传播逻辑）（也就是model定义了代码实现）
    # litmodel 通常是指使用 PyTorch Lightning 库创建的模型类。PyTorch Lightning 是一个高级库，旨在简化复杂的模型训练过程。litmodel 类通常封装了一个基础模型（比如 bert-large-uncased 或 roberta），并添加了训练、验证和测试循环的逻辑，以及其他可能需要的步骤，比如优化器和学习率调度器的配置。简单来说，litmodel 更多地关注于如何使用基础模型进行有效的训练和验证

    parser.add_argument("--two_steps", default=False, action="store_true")  
    # 单阶段训练在整个数据集上进行训练，包括前向传播，反向传播，和参数更新
    # 两阶段训练通常是在一个数据集上进行预训练，第二阶段是在另一个数据集上进行微调或者进一步的训练，通常用于迁移学习或者领域自适应的任务中，即模型首先从一个数据集学习通用特征，然后再另一个数据集上进行特定任务的微调
    parser.add_argument("--load_checkpoint", type=str, default=None)  # 指定要加载的模型的模型检查点文件的路径，模型检查点文件通常包含了已经训练好的模型的权重以及其它必要的信息，加载模型检查点文件，那么模型将使用这个文件中保存的模型状态来继续进行训练或进行其它任务

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")  # 创建一个名为“Data Args”的参数组，并将这个参数组赋值给data_group
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"
from tqdm import tqdm
def _get_relation_embedding(data):
    train_dataloader = data.train_dataloader()
    #! hard coded
    relation_embedding = [[] for _ in range(36)]   # ？？？？？？？？ 应该根据数据集的格式而设置，relation_embedding用于存放每个样本【103】（分类头）这个token对应的logits，所以shape：【36，hidden—_size】,36对应关系类型个数
    model = AutoModel.from_pretrained('bert-base-uncased')  # 这个模型也需要自己制定
    model.eval()
    model = model.to(device)


    cnt = 0
    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            #! why the sample in this case will cause errors
            if cnt == 416:
                continue
            cnt += 1
            input_ids, attention_mask, token_type_ids , labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state.detach().cpu()  # 使用模型对输入文本进行编码（前向计算），并获取模型的输出logits，这个输出包含了对输入文本的表示，最后一个隐藏状态通常包含了对输入文本的编码信息，可以用于进一步的任务或分析
            _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)  # [SEP] 103 ,第二个返回值 (mask_idx): 这个变量包含了 True 元素在第二个维度（例如，在二维数组中是列）的索引。在这个场景中，因为我们关心的是等于 103 的元素的位置，mask_idx 就是我们感兴趣的索引集
            bs = input_ids.shape[0]
            mask_output = logits[torch.arange(bs), mask_idx] # [batch_size, hidden_size] ，将坐标（batch_id,SEP_id）的元素logits取出
            

            labels = labels.detach().cpu()  
            mask_output = mask_output.detach().cpu()
            assert len(labels[0]) == len(relation_embedding)  # labels：（batchsize，hiddensize）,每个样本的关系标签概率分布，relation_embedding：（36,1） 
            for batch_idx, label in enumerate(labels.tolist()):
                for i, x in enumerate(label):  # label：【0,1,0，0,0,0】，len=36，对应真实样本的关系类型概率分布
                    if x:
                        relation_embedding[i].append(mask_output[batch_idx]) 
    
    # get the mean pooling
    for i in range(36):  # 遍历每个样本的relation_embedding，即关系的预测logits，维度为（36，36）
        if len(relation_embedding[i]): # 这个关系类型里面有样本sep对应的logits
            relation_embedding[i] = torch.mean(torch.stack(relation_embedding[i]), dim=0)
        else: # 这个关系类型里面没有样本sep对应的logits
            relation_embedding[i] = torch.rand_like(relation_embedding[i-1])

    del model
    return relation_embedding

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}") # <class 'data.dialogue.WIKI80'>
    model_class = _import_class(f"models.{args.model_class}")  # args.model_class：'RobertaForPrompt'， model_class：<class 'models.RobertaForPrompt'>
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}") # args.model_name_or_path：'bert-base-chinese'，litmodel_class： <class 'lit_models.transformer.BertLitModel'>

    config = AutoConfig.from_pretrained(args.model_name_or_path)   # 加载配置文件
    model = model_class.from_pretrained(args.model_name_or_path, config=config)  # 加载模型文件
    data = data_class(args, model)  # 
    data_config = data.get_data_config()  # data_config: {'num_labels': 12} 12种关系
    model.resize_token_embeddings(len(data.tokenizer))

    

    # gpt no config?

    # if "gpt" in args.model_name_or_path or "roberta" in args.model_name_or_path:
    #     tokenizer = data.get_tokenizer()
    #     model.resize_token_embeddings(len(tokenizer))
    #     model.update_word_idx(len(tokenizer))
    #     if "Use" in args.model_class:
    #         continous_prompt = [a[0] for a in tokenizer([f"[T{i}]" for i in range(1,3)], add_special_tokens=False)['input_ids']]
    #         continous_label_word = [a[0] for a in tokenizer([f"[class{i}]" for i in range(1, data.num_labels+1)], add_special_tokens=False)['input_ids']]
    #         discrete_prompt = [a[0] for a in tokenizer(['It', 'was'], add_special_tokens=False)['input_ids']]
    #         dataset_name = args.data_dir.split("/")[1]
    #         model.init_unused_weights(continous_prompt, continous_label_word, discrete_prompt, label_path=f"{args.model_name_or_path}_{dataset_name}.pt")
    # data.setup()
    # relation_embedding = _get_relation_embedding(data)
    # BertLitMode
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)  # 根据参数实例化lit model
    data.tokenizer.save_pretrained('test')  # 保存分词器到本地目录test
    print(lit_model)


    logger = pl.loggers.TensorBoardLogger("training/logs")  # pl.loggers.TensorBoardLogger 是 PyTorch Lightning 提供的一个日志记录器类，用于将训练过程中的日志信息记录到 TensorBoard 格式的日志文件中
    dataset_name = args.data_dir.split("/")[-1]
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")  # pl.loggers.WandbLogger 是 PyTorch Lightning 提供的一个日志记录器类，用于将训练过程中的日志信息记录到 Weights and Biases (WandB) 平台上
        logger.log_hyperparams(vars(args))
    
    # init callbacks
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5, check_on_train_epoch_end=False)  # early_callback 是一个用于提前停止训练的回调对象，通常用于避免过拟合或在训练不再改善时节省计算资源
    '''
    monitor="Eval/f1"：这是回调函数监视的指标名称。EarlyStopping 会根据这个指标的值来决定是否停止训练。在这里，它监视 "Eval/f1" 指标，通常是验证集上的 F1 分数（F1分数：在深度学习模型训练过程中一种用于评估模型性能的指标。它通常用于分类任务，特别是二分类任务，例如情感分析、文本分类、图像分类。在训练过程中，模型会在每个训练周期（或称为 epoch）结束后进行验证，以评估其在验证集上的性能。"F1 分数" 是一个综合性能指标，结合了模型的精确度和召回率。计算公式为：(2x查准率x查全率)/(查准率+查全率)。通常情况下，F1 分数的取值范围在 0 到 1 之间，越接近 1 表示模型性能越好。）
    mode="max"：这是模式参数，指定了如何确定是否达到停止条件。"max" 意味着当监视的指标达到最大值时停止训练。如果你希望在监视指标达到最小值时停止训练，可以将 mode 设置为 "min"

    patience=5：这是容忍参数，表示在监视指标没有改善的情况下，等待多少个训练周期后才停止训练。在这里，如果 "Eval/f1" 指标在连续 5 个训练周期中没有改善，则训练将停止

    check_on_train_epoch_end=False：这个参数表示是否在每个训练周期结束时检查监视指标。如果设置为 True，则在每个训练周期结束后都会检查一次，如果满足停止条件，就会停止训练。在这里，它设置为 False，表示只在验证集上的评估周期结束时检查监视指标

    通过将这个提前停止的回调对象传递给 PyTorch Lightning 的训练器（Trainer），你可以在训练过程中启用提前停止功能，以便在验证集上的性能不再改善时自动停止训练，从而避免过拟合
    '''
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )
    '''
    model_checkpoint 是一个用于保存模型检查点的回调对象，通常用于在训练过程中保存模型的权重或整个模型，以便在训练结束后或需要恢复训练时使用

    具体来说，这行代码使用 PyTorch Lightning 提供的 ModelCheckpoint 回调类创建了一个名为 model_checkpoint 的回调对象，其配置如下：

    monitor="Eval/f1"：这是回调函数监视的指标名称。ModelCheckpoint 会根据这个指标的值来决定是否保存模型检查点。在这里，它监视 "Eval/f1" 指标，通常是验证集上的 F1 分数

    mode="max"：这是模式参数，指定了如何确定是否保存模型检查点。"max" 意味着当监视的指标达到最大值时保存检查点。如果你希望在监视指标达到最小值时保存检查点，可以将 mode 设置为 "min"

    filename='{epoch}-{Eval/f1:.2f}'：这是保存模型检查点的文件名模板，其中 {epoch} 表示当前训练周期的编号，{Eval/f1:.2f} 表示当前 "Eval/f1" 指标的值，保留两位小数。这将在每个训练周期结束时生成一个唯一的文件名，以保存模型检查点

    dirpath="output"：这是保存模型检查点的文件夹路径。模型检查点将保存在名为 "output" 的文件夹中。你可以根据需要更改保存路径

    save_weights_only=True：这个参数表示是否仅保存模型的权重而不保存整个模型。如果设置为 True，则只保存模型权重；如果设置为 False，则保存整个模型。在这里，它设置为 True，表示只保存权重

    通过将这个模型检查点的回调对象传递给 PyTorch Lightning 的训练器（Trainer），你可以在训练过程中启用模型检查点功能，以便在每个训练周期结束时根据指定的条件保存模型的权重或整个模型。这对于在训练中定期保存模型、避免训练中断或用于后续评估和推理非常有用

    '''
    callbacks = [early_callback, model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None
    '''
    根据GPU数量选择适当的训练加速器

    如果 gpu_count > 1，即系统中有多个 GPU 可用，那么 accelerator 被设置为 "ddp"，表示使用分布式数据并行（Distributed Data Parallel）加速器。Distributed Data Parallel 允许在多个 GPU 上并行训练模型，以加速训练过程

    如果 gpu_count <= 1，即系统中只有一个或没有 GPU 可用，那么 accelerator 被设置为 None，表示不使用任何特殊的分布式加速器，而是在单个 GPU 或 CPU 上进行训练

    '''


    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
        plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
    )
    '''
    创建 PyTorch Lightning 的训练器（Trainer）对象，并配置训练器的各种参数，以便进行模型训练

    pl.Trainer.from_argparse_args(args, ...)：通过调用 from_argparse_args 方法，从命令行参数 args 中加载训练器的配置参数。这是一种方便的方式，允许从命令行传递参数来配置训练器

    default_root_dir="training/logs"：指定模型训练过程中保存日志和检查点的根目录

    plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None：这是一个插件配置，用于根据 GPU 数量选择是否启用 Distributed Data Parallel（"ddp"）插件。如果 gpu_count > 1，则启用 "ddp" 插件，并设置 find_unused_parameters=False，表示不查找未使用的参数

    我们在使用 PyTorch Lightning 的 Trainer 时，已经在 accelerator 参数中设置了正确的分布式加速器（即 "ddp"），所以实际上不需要显式添加 DDPPlugin 插件。accelerator 参数的设置已经包括了 DDP 训练的配置，会自动启用 DDP 并管理 GPU 上的并行训练

    '''

    # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)  # 开始真正的训练和验证模型
    '''
    这行代码用于开始模型的训练过程

    lit_model：这是你要训练的 PyTorch Lightning 模型，通常LightningModule 的子类，包含模型的定义和训练逻辑

    datamodule=data：这是数据模块，用于提供训练和验证数据集以及数据加载器
    trainer.fit 方法将使用指定的模型和数据模块来执行训练循环，包括多个训练周期（epochs），每个周期包括数据加载、前向传播、反向传播、参数更新等步骤。具体来说，它会执行以下操作：
    '''

    # two steps

    path = model_checkpoint.best_model_path
    print(f"best model save path {path}")

    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
    config = vars(args)
    config["path"] = path
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

    # lit_model.load_state_dict(torch.load(path)["state_dict"])

    if not args.two_steps: trainer.test()
    '''
    trainer.test() 是 PyTorch Lightning 中的一个方法，用于在训练结束后运行模型的测试阶段。在测试阶段，模型将被用来评估其在测试数据集上的性能，通常用于生成测试结果、计算指标（如准确率、F1 分数等）或者进行其他与模型性能评估相关的操作

    具体来说，trainer.test() 的功能包括：将模型切换到评估模式、在测试数据集上运行模型，获取模型的输出、计算并记录各种性能指标（如损失、准确率、F1 分数等）、打印或记录测试结果
    '''
    step2_model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
        filename='{epoch}-{Step2Eval/f1:.2f}',
        dirpath="output",
        save_weights_only=True
    )

    if args.two_steps:
        # we build another trainer and model for the second training
        # use the Step2Eval/f1 

        # lit_model_second = TransformerLitModelTwoSteps(args=args, model=lit_model.model, data_config=data_config)
        step_early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=6, check_on_train_epoch_end=False)
        callbacks = [step_early_callback, step2_model_checkpoint]
        trainer_2 = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        )
        trainer_2.fit(lit_model, datamodule=data)
        trainer_2.test()
        # result = trainer_2.test(lit_model, datamodule=data)[0]
        # with open("result.txt", "a") as file:
        #     a = result["Step2Test/f1"]
        #     file.write(f"test f1 score: {a}\n")
        #     file.write(config_file_name + '\n')

    # trainer.test(datamodule=data)


if __name__ == "__main__":

    main()
