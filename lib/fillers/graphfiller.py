import torch
from . import Filler

class GraphFiller(Filler):

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        super(GraphFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        self.trimming = (warm_up, warm_up)
    
    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        float_mask =  batch_data['mask'].clone().detach() 
        tra_mask = torch.where(float_mask == 1, torch.ones_like(batch_data['mask']), torch.zeros_like(batch_data['mask']))
        float_part_mask = float_mask - tra_mask
        white_tra_mask = torch.bernoulli(tra_mask.clone().detach().float() * self.keep_prob).byte()
        white_float_mask = white_tra_mask + float_part_mask 
        eval_mask = batch_data.pop('eval_mask', None)
        ori_eval_mask = eval_mask
        eval_mask = eval_mask + tra_mask - white_tra_mask
        mask = 1 - ori_eval_mask
        y = batch_data.pop('y')
        batch_data['mask'] = white_float_mask
        loss_mask = tra_mask
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions, _ = (res[0], res[1], res[2]) if isinstance(res, (list, tuple)) else (res, [])
        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)
        loss = self.loss_fn(imputation, target, loss_mask.byte()) 
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, loss_mask.byte()) 
        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None) 
        y = batch_data.pop('y') 
        # Compute predictions and compute loss
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
        val_loss = self.loss_fn(imputation, target, eval_mask.byte())
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask.byte())
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask_ = batch_data.pop('eval_mask', None)
        ori_eval_mask = eval_mask_ 
        eval_mask = 1 - mask - eval_mask_ 
        y = batch_data.pop('y')
        mask_ori = mask 
        mask = 1 - ori_eval_mask
        batch_data['mask'] = ori_eval_mask + mask_ori
        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)
        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss