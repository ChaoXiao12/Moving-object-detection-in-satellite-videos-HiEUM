from lib.Trainer.ctdet_normal import CtdetTrainer_normal
from lib.Trainer.ctdet_points import CtdetTrainer_points

trainer_factory = {
    'ctdet_normal':CtdetTrainer_normal,
    'ctdet_points':CtdetTrainer_points,
}

def get_trainer(opt, model, optimizer):
    trainer = trainer_factory[opt.task](opt, model, optimizer)
    return trainer