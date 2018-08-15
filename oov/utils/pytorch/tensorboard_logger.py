# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import os


class TensorboardLogger(object):
    def __init__(self, log_dir, start_step=0):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        if start_step != 0:
            self.writer.add_session_log(
                tf.SessionLog(status=tf.SessionLog.START),
                global_step=start_step)

    def scalar_summary(self, scalar_dict, step):
        """Log a scalar variable."""
        values = [tf.Summary.Value(tag=tag, simple_value=value) for tag, value
                  in scalar_dict.items()]
        summary = tf.Summary(value=values)
        self.writer.add_summary(summary, step)

    @classmethod
    def from_dir(cls, log_dir, split, start_step=0):
        # Create log_dir if it does not exist
        if not os.path.exists(log_dir):
            os.makedirs(os.path.join(log_dir, split))
        return cls(os.path.join(log_dir, split), start_step)
