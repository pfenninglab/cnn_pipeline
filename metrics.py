import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow_addons.metrics
import numpy as np


class MulticlassAUC(tensorflow.keras.metrics.AUC):
    # adapted from https://stackoverflow.com/a/63604257
    def __init__(self, pos_label, from_logits=False, sparse=True, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for the metric's update_state() method.
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_label": self.pos_label,
            "from_logits": self.from_logits,
            "sparse": self.sparse
        })
        return config

class MulticlassMetric(tensorflow.keras.metrics.Metric):
    """Binary metric for a multiclass problem, by treating one label as positive and the rest as negative.
    adapted from https://stackoverflow.com/a/63604257

    This implementation allows you to plug in any binary Keras metric.

    Args:
        k_metric_name (str): Keras metric class name as a string.
        pos_label : int
            Label of the positive class (the one whose metric is being computed).

        from_logits : bool, optional (default: False)
            If True, assume predictions are not standardized to be between 0 and 1.
            In this case, predictions will be squeezed into probabilities using the
            softmax function.

        sparse : bool, optional (default: True)
            If True, ground truth labels should be encoded as integer indices in the
            range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
            encoded indicator vectors (with a 1 in the true label position and 0
            elsewhere).

        make_dense: bool, optional (default: False)
            if False, then convert y_true and y_pred to sparse shape (batch_size,) before
                passing to metric
            if True, then convert y_true and y_pred to dense shape (batch_size, num_classes)

        **kwargs : keyword arguments
            Keyword arguments to be passed to Keras metric.

    """
    def __init__(self, k_metric_name, pos_label, from_logits=False, sparse=True, make_dense=False, **kwargs):
        super().__init__(name=kwargs['name'])
        self.k_metric_name = k_metric_name
        self.k_metric = self._get_k_metric(**kwargs)
        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse
        self.make_dense = make_dense

    def update_state(self, y_true, y_pred, **kwargs):
        """Accumulates confusion matrix statistics.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).

        y_pred : tf.Tensor
            The predicted values, a tensor of shape (n_examples, n_classes).

        **kwargs : keyword arguments
            Extra keyword arguments for the metric's update_state() method.
        """
        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]
        if self.make_dense:
            num_classes = y_pred.shape[1]
            y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_classes)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        if not self.make_dense:
            y_pred = y_pred[..., self.pos_label]

        self.k_metric.update_state(y_true, y_pred, **kwargs)

    def result(self):
        res = self.k_metric.result()
        if self.make_dense:
            # Assumes that this metric returns one value for each class.
            # Return the value for the positive class.
            res = res[self.pos_label]
        return res

    def reset_state(self):
        self.k_metric.reset_state()

    def get_config(self):
        """For model saving and loading"""
        config = self.k_metric.get_config()
        config.update({
            "k_metric_name": self.k_metric_name,
            "pos_label": self.pos_label,
            "from_logits": self.from_logits,
            "sparse": self.sparse,
            "make_dense": self.make_dense
        })
        return config

    def _get_k_metric(self, **kwargs):
        # Search for a metric with this name in:
        # 1. Tensorflow keras metrics
        # 2. Tensorflow Addons metrics
        # 3. Custom metrics from this module
        current_module = __import__(__name__)
        for module in [tensorflow.keras.metrics, tensorflow_addons.metrics, current_module]:
            k_metric = getattr(module, self.k_metric_name, None)
            if k_metric is not None:
                break

        if k_metric is None:
            raise ValueError(f"Could not find keras metric {self.k_metric_name}")
        return k_metric(**kwargs)


class PearsonCorrelation(tensorflow.keras.metrics.Metric):
    """Pearson correlation coefficient metric for TensorFlow.
    
    This metric computes the Pearson correlation coefficient between true and predicted values.
    It works for both regression and classification problems (using predicted probabilities).
    """
    
    def __init__(self, name='pearson_correlation', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_x = self.add_weight(name='sum_x', initializer='zeros')
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros')
        self.sum_x_squared = self.add_weight(name='sum_x_squared', initializer='zeros')
        self.sum_y_squared = self.add_weight(name='sum_y_squared', initializer='zeros')
        self.sum_xy = self.add_weight(name='sum_xy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the correlation statistics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional weighting of samples
        """
        # Handle classification case - use predicted probabilities for positive class
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            # For classification, use the probability of the positive class
            y_pred = y_pred[..., -1]  # Take the last column (positive class probability)
        
        # Flatten tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1])
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight
            weight_sum = tf.reduce_sum(sample_weight)
        else:
            weight_sum = tf.cast(tf.size(y_true), tf.float32)
        
        # Update running statistics
        self.sum_x.assign_add(tf.reduce_sum(y_true))
        self.sum_y.assign_add(tf.reduce_sum(y_pred))
        self.sum_x_squared.assign_add(tf.reduce_sum(y_true * y_true))
        self.sum_y_squared.assign_add(tf.reduce_sum(y_pred * y_pred))
        self.sum_xy.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(weight_sum)

    def result(self):
        """Compute the Pearson correlation coefficient."""
        # Avoid division by zero
        count = tf.maximum(self.count, 1.0)
        
        # Compute means
        mean_x = self.sum_x / count
        mean_y = self.sum_y / count
        
        # Compute correlation coefficient
        numerator = self.sum_xy / count - mean_x * mean_y
        denominator_x = tf.sqrt(tf.maximum(self.sum_x_squared / count - mean_x * mean_x, 1e-8))
        denominator_y = tf.sqrt(tf.maximum(self.sum_y_squared / count - mean_y * mean_y, 1e-8))
        
        correlation = numerator / (denominator_x * denominator_y)
        return correlation

    def reset_state(self):
        """Reset all statistics."""
        self.sum_x.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_x_squared.assign(0.0)
        self.sum_y_squared.assign(0.0)
        self.sum_xy.assign(0.0)
        self.count.assign(0.0)


class SpearmanCorrelation(tensorflow.keras.metrics.Metric):
    """TensorFlow implementation of Spearman rank correlation coefficient.
    
    Note: This is an approximation that computes correlation on batch-level ranks.
    For exact Spearman correlation across the entire dataset, consider using
    a callback that computes it at the end of each epoch.
    """

    def __init__(self, name='spearman_correlation', **kwargs):
        super().__init__(name=name, **kwargs)
        # Store running statistics for batch-level correlation
        self.sum_x = self.add_weight(name='sum_x', initializer='zeros')
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros')
        self.sum_x_sq = self.add_weight(name='sum_x_sq', initializer='zeros')
        self.sum_y_sq = self.add_weight(name='sum_y_sq', initializer='zeros')
        self.sum_xy = self.add_weight(name='sum_xy', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # For Spearman correlation, we need to compute ranks
        # Since we can't store all values across batches in graph mode,
        # we'll use a batch-level approximation
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Compute ranks within this batch
        y_true_rank = tf.cast(tf.argsort(tf.argsort(y_true_flat)), tf.float32)
        y_pred_rank = tf.cast(tf.argsort(tf.argsort(y_pred_flat)), tf.float32)
        
        # Normalize ranks to [0, 1] range for better batch-level correlation
        batch_size = tf.cast(tf.shape(y_true_rank)[0], tf.float32)
        y_true_rank = y_true_rank / tf.maximum(batch_size - 1, 1)
        y_pred_rank = y_pred_rank / tf.maximum(batch_size - 1, 1)
        
        # Update running sums
        self.sum_x.assign_add(tf.reduce_sum(y_true_rank))
        self.sum_y.assign_add(tf.reduce_sum(y_pred_rank))
        self.sum_x_sq.assign_add(tf.reduce_sum(tf.square(y_true_rank)))
        self.sum_y_sq.assign_add(tf.reduce_sum(tf.square(y_pred_rank)))
        self.sum_xy.assign_add(tf.reduce_sum(y_true_rank * y_pred_rank))
        self.count.assign_add(batch_size)

    def result(self):
        n = self.count
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denominator_x = n * self.sum_x_sq - tf.square(self.sum_x)
        denominator_y = n * self.sum_y_sq - tf.square(self.sum_y)
        denominator = tf.sqrt(denominator_x * denominator_y)
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        self.sum_x.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_x_sq.assign(0.0)
        self.sum_y_sq.assign(0.0)
        self.sum_xy.assign(0.0)
        self.count.assign(0.0)
