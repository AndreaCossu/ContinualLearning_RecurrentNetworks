'''
https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset
'''

import torch
import clutils
from clutils import monitors
from clutils.experiments import Trainer
from clutils.metrics import accuracy
from clutils.strategies import EWC, MAS, Rehearsal, LWF, GEM, AGEM


def words_exp(args):
    N_MELS = 40

    classes = [ 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
                'house', 'left', 'marvel', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

    if args.multitask:
        args.n_tasks = 1
        args.output_size = len(classes)
        classes = [classes]
    else:
        if hasattr(args, 'model_selection'):
            if args.model_selection: # used for model selection
                classes = classes[:10]
            else: # used for model assessment
                classes = classes[10:]
        classes = [ [classes[i+j] for j in range(args.classes_per_task)] for i in range(0,len(classes),args.classes_per_task)]

    if args.agem: # prepare for 1-hot task input vector
        args.input_size += args.n_tasks

    device = clutils.experiments.utils.get_device(args.cuda)
    result_folder = clutils.experiments.utils.create_result_folder(args.result_folder)
    logger = monitors.log.create_logger(result_folder)
    metric_monitor = monitors.log.LogMetric(args.n_tasks, result_folder, 'acc')

    monitors.log.write_configuration(args, result_folder)

    model = clutils.experiments.utils.create_models(args, device)
    optimizer = clutils.experiments.utils.create_optimizers(model, args.learning_rate, args.weight_decay, args.use_sgd)
    criterion = torch.nn.CrossEntropyLoss()
    model, optimizer = model[args.models], optimizer[args.models]

    if model.is_recurrent:
        model.output_type = clutils.OUTPUT_TYPE.LAST_OUT
    else:
        model.output_type = clutils.OUTPUT_TYPE.ALL_OUTS

    # It is possible to add: penalties={'l1': 1e-4}) to trainer
    trainer = Trainer(model, optimizer, criterion, device, accuracy, clip_grad=args.clip_grad)

    if args.monitor:
        writer = monitors.plots.create_writer(result_folder)

    ###################################################################################
    ###################################################################################
    ###################################################################################

    if args.ewc_lambda > 0:
        ewc = EWC(model, device, args.ewc_lambda, normalize=True, single_batch=False, cumulative='none')
    if args.mas_lambda > 0:
        mas = MAS(model, device, args.mas_lambda)
    if args.gem:
        gem = GEM(args.gem_patterns_per_step, args.gem_memory_strength)
    if args.agem:
        agem = AGEM(args.gem_patterns_per_step, args.agem_sample_size)
    if args.rehe_patterns > 0:
        rehe = Rehearsal(args.rehe_patterns, args.patterns_per_class_per_batch)
    if args.lwf:
        args.lwf_alpha = args.lwf_alpha[0] if isinstance(args.lwf_alpha, (list, tuple)) and len(args.lwf_alpha) == 1 else args.lwf_alpha
        lwf = LWF(model, device, classes_per_task=args.classes_per_task, alpha=args.lwf_alpha,
            temperature=args.lwf_temp, warmup_epochs=args.warmup_epochs)

    if args.save:
        clutils.experiments.utils.save_model(model, args.models, result_folder, version='_init')

    dataset = clutils.datasets.CLSpeechWords(args.dataroot, perc_test=0.2, train_batch_size=args.batch_size,
            test_batch_size=256, n_mels=N_MELS,  len_task_vector=args.n_tasks if args.agem else 0,
            task_vector_at_test=args.task_vector_at_test, return_sequences=model.is_recurrent)

        
    logger.warning("model,task_id,epoch,train_acc,validation_acc,train_loss,validation_loss")

    compute_for_head = True
    freeze_after_first = False

    if not args.load:
        for task_id in range(args.n_tasks):

            if task_id == 1 and freeze_after_first:
                for k,p in model.named_parameters():
                    if k != 'layers.out.bias' and k!= 'layers.out.weight':
                        p.requires_grad = False


            loader_task_train, loader_task_test = dataset.get_task_loaders(classes=classes[task_id])

            if args.warmup_epochs > 0:
                lwf.warmup(loader_task_train, task_id)

            ######## VALIDATION BEFORE TRAINING ########
            for x,y in loader_task_test:
                x,y = x.to(device), y.to(device)

                validation_loss, validation_accuracy = trainer.test(x,y, task_id=(task_id if args.multihead else None))
                metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

            metric_monitor.update_metrics(args.models, 'val', task_id, num_batches=len(loader_task_test), reset_averages=True)
            logger.warning(
                f"{args.models},"
                f"{task_id},0,"
                f"-1,"
                f"{metric_monitor.get_metric(args.models, 'val', 'acc', task_id)},"
                f"-1,"
                f"{metric_monitor.get_metric(args.models, 'val', 'loss', task_id)}"
            )

            ######## START SPECIFIC TASK ########

            if args.rehe_patterns > 0:
                loader_train = rehe.augment_dataset(loader_task_train)
            else:
                loader_train = loader_task_train

            for epoch in range(1, args.epochs+1):
                logger.info(f"Task {task_id} - Epoch {epoch}/{args.epochs}")

                ######## TRAINING ########

                for id_batch, (x,y) in enumerate(loader_train):

                    if args.rehe_patterns > 0:
                        x, y = rehe.concat_to_batch(x,y)

                    x,y = x.to(device), y.to(device)

                    if args.ewc_lambda > 0:
                        training_loss, training_accuracy = trainer.train_ewc(x, y, ewc, task_id, multi_head=args.multihead)            
                    elif args.mas_lambda > 0:
                        training_loss, training_accuracy = trainer.train_mas(x, y, mas, task_id, args.truncated_time, multi_head=args.multihead)
                    elif args.lwf:
                        training_loss, training_accuracy = trainer.train_lwf(x, y, lwf, task_id)
                    elif args.gem:
                        training_loss, training_accuracy = trainer.train_gem(x, y, gem, task_id)
                    elif args.agem:
                        training_loss, training_accuracy = trainer.train_agem(x, y, agem, task_id, multi_head=args.multihead)
                    else:
                        training_loss, training_accuracy = trainer.train(x,y, task_id=(task_id if args.multihead else None))

                    metric_monitor.update_averages(args.models, training_loss, training_accuracy)
                    
                metric_monitor.update_metrics(args.models, 'train', task_id, num_batches=len(loader_train), reset_averages=True)

                ######## VALIDATION ########

                for x,y in loader_task_test:
                    x,y = x.to(device), y.to(device)

                    validation_loss, validation_accuracy = trainer.test(x,y, task_id=(task_id if args.multihead else None))
                    metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

                metric_monitor.update_metrics(args.models, 'val', task_id, num_batches=len(loader_task_test), reset_averages=True)
                logger.warning(
                    f"{args.models},"
                    f"{task_id},{epoch}," 
                    f"{metric_monitor.get_metric(args.models, 'train', 'acc', task_id)},"
                    f"{metric_monitor.get_metric(args.models, 'val', 'acc', task_id)},"
                    f"{metric_monitor.get_metric(args.models, 'train', 'loss', task_id)},"
                    f"{metric_monitor.get_metric(args.models, 'val', 'loss', task_id)}"
                )                

            ######## END OF CURRENT TASK ########
            if args.save:
                clutils.experiments.utils.save_model(model, args.models, result_folder, version=str(task_id))

            if args.monitor:
                monitors.plots.plot_weights(writer, args.models, model, task_id, epoch)
                # monitors.plots.plot_gradients(writer, args.models, model, task_id, epoch)

            if args.ewc_lambda > 0:
                f, unnorm_f = ewc.compute_importance(optimizer, criterion, task_id, 
                    loader_task_train, update=True, truncated_time=args.truncated_time,
                    compute_for_head=compute_for_head)
                if args.monitor:
                    monitors.plots.plot_importance(writer, args.models, f, task_id)

            if args.mas_lambda > 0:
                mas.update_importance(task_id, mas.current_importance, save_pars=True)
                mas.reset_current_importance()
                if args.monitor:
                    monitors.plots.plot_importance(writer, args.models, mas.importance[task_id], task_id)

            if args.gem:
                gem.update_memory(loader_task_train, task_id)

            if args.agem:
                agem.update_memory(loader_task_train)

            if args.lwf:
                lwf.save_model()


            if task_id < args.n_tasks - 1:

                if args.expand_output > 0:
                    model.expand_output_layer(n_units=args.expand_output)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                    trainer.optimizer = optimizer

                if args.rehe_patterns > 0:
                    rehe.record_patterns(loader_task_train)

            ######## INTERMEDIATE TEST ########
            # DO THIS TEST FOR ALL PREVIOUS TASKS, IF ARGS.NOT_INTERMEDIATE_TEST IS DISABLED
            # OR FOR THE LAST TASK ONLY, IF ARGS.NOT_TEST IS DISABLED

            if not args.not_intermediate_test or ( (not args.not_test) and (task_id == args.n_tasks-1) ):
                for intermediate_task_id in range(task_id+1):

                    _, loader_task_test = dataset.get_task_loaders(task_id=intermediate_task_id)

                    for x,y in loader_task_test:
                        x,y = x.to(device), y.to(device)

                        intermediate_loss, intermediate_accuracy = trainer.test(x,y, task_id=(intermediate_task_id if args.multihead else None))
                        metric_monitor.update_averages(args.models, intermediate_loss, intermediate_accuracy)

                    metric_monitor.update_intermediate_metrics(args.models, len(loader_task_test), task_id, intermediate_task_id)
                    logger.info(f"Intermediate accuracy {args.models}:"
                        f"{task_id} - {intermediate_task_id}:"
                        f"{metric_monitor.intermediate_metrics[args.models]['acc'][intermediate_task_id, task_id]}")


    if args.monitor:
        writer.close()

    logger.info("Saving results...")
    metric_monitor.save_intermediate_metrics()
    monitors.plots.plot_learning_curves(args.models, result_folder)
    logger.info("Done!")


def words_multitask_exp(args):
    N_MELS = 40

    classes = [ #'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
                'house', 'left', 'marvel', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

    args.n_tasks = 1

    device = clutils.experiments.utils.get_device(args.cuda)
    result_folder = clutils.experiments.utils.create_result_folder(args.result_folder)
    logger = monitors.log.create_logger(result_folder)
    metric_monitor = monitors.log.LogMetric(args.n_tasks, result_folder, 'acc')

    monitors.log.write_configuration(args, result_folder)

    model = clutils.experiments.utils.create_models(args, device)
    optimizer = clutils.experiments.utils.create_optimizers(model, args.learning_rate, args.weight_decay, args.use_sgd)
    criterion = torch.nn.CrossEntropyLoss()
    model, optimizer = model[args.models], optimizer[args.models]

    if model.is_recurrent:
        model.output_type = clutils.OUTPUT_TYPE.LAST_OUT
    else:
        model.output_type = clutils.OUTPUT_TYPE.ALL_OUTS

    # It is possible to add: penalties={'l1': 1e-4}) to trainer
    trainer = Trainer(model, optimizer, criterion, device, accuracy, clip_grad=args.clip_grad)

    ###################################################################################
    ###################################################################################
    ###################################################################################

    if args.save:
        clutils.experiments.utils.save_model(model, args.models, result_folder, version='_init')

    dataset = clutils.datasets.SpeechWords(args.dataroot, perc_test=0.15, train_batch_size=args.batch_size,
            test_batch_size=256, n_mels=N_MELS, return_sequences=model.is_recurrent)


    logger.warning("model,task_id,epoch,train_acc,validation_acc,train_loss,validation_loss")

    loader_train, loader_val, loader_test = dataset.get_loaders(classes)
    loader_test = loader_val if hasattr(args, 'model_selection') and args.model_selection else loader_test

    ######## VALIDATION BEFORE TRAINING ########
    for x,y in loader_test:
        x,y = x.to(device), y.to(device)

        validation_loss, validation_accuracy = trainer.test(x,y)
        metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

    metric_monitor.update_metrics(args.models, 'val', 0, num_batches=len(loader_test), reset_averages=True)
    logger.warning(
        f"{args.models},"
        f"0,0,"
        f"-1,"
        f"{metric_monitor.get_metric(args.models, 'val', 'acc', 0)},"
        f"-1,"
        f"{metric_monitor.get_metric(args.models, 'val', 'loss', 0)}"
    )

    ######## START SPECIFIC TASK ########

    for epoch in range(1, args.epochs+1):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        ######## TRAINING ########

        for id_batch, (x,y) in enumerate(loader_train):

            x,y = x.to(device), y.to(device)

            training_loss, training_accuracy = trainer.train(x,y)

            metric_monitor.update_averages(args.models, training_loss, training_accuracy)

        metric_monitor.update_metrics(args.models, 'train', 0, num_batches=len(loader_train), reset_averages=True)

        ######## VALIDATION ########

        for x,y in loader_test:
            x,y = x.to(device), y.to(device)

            validation_loss, validation_accuracy = trainer.test(x,y)
            metric_monitor.update_averages(args.models, validation_loss, validation_accuracy)

        if epoch == args.epochs:
            metric_monitor.update_intermediate_metrics(args.models, len(loader_test), 0, 0, reset_averages=False)
        metric_monitor.update_metrics(args.models, 'val', 0, num_batches=len(loader_test), reset_averages=True)
        logger.warning(
            f"{args.models},"
            f"0,{epoch}," 
            f"{metric_monitor.get_metric(args.models, 'train', 'acc', 0)},"
            f"{metric_monitor.get_metric(args.models, 'val', 'acc', 0)},"
            f"{metric_monitor.get_metric(args.models, 'train', 'loss', 0)},"
            f"{metric_monitor.get_metric(args.models, 'val', 'loss', 0)}"
        )

    if args.save:
        clutils.experiments.utils.save_model(model, args.models, result_folder, version='0')

    logger.info("Saving results...")
    metric_monitor.save_intermediate_metrics()
    monitors.plots.plot_learning_curves(args.models, result_folder)
    logger.info("Done!")
