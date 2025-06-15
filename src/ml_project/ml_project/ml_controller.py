import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Fruit handlers
from ml_project.datasets.fruit import (
    load_and_featurize as fruit_load,
    visualize as fruit_visualize,
    train as fruit_train,
    evaluate as fruit_evaluate,
    feature_importance as fruit_importance,
)

# Iris handlers
from ml_project.datasets.iris import (
    load_and_featurize as iris_load,
    visualize as iris_visualize,
    train as iris_train,
    evaluate as iris_evaluate,
    feature_importance as iris_importance,
)

# Cancer & Penguin handlers
from ml_project.datasets.cancer_penguin import (
    load_and_featurize_cancer,
    load_and_featurize_penguin,
    visualize_both as cp_visualize,
    train_cancer as cp_train_cancer,
    train_penguin as cp_train_penguin,
    evaluate_both as cp_evaluate_both,
    feature_importance_both as cp_importance_both,
)

class MLControllerNode(Node):
    def __init__(self):
        super().__init__('ml_controller')
        self.get_logger().info("ML Controller ready. Publish commands to /ml_command")
        self.create_subscription(String, '/ml_command', self.on_command, 10)

        # fruit state
        self.df_f = self.X_f = self.y_f = None
        self.model_f = self.holdout_f = None

        # iris state
        self.df_i = self.X_i = self.y_i = None
        self.model_i = self.holdout_i = None
        self.iris_model_name = None

        # cancer_penguin state
        self.df_c = self.X_c = self.y_c = None
        self.df_p = self.X_p = self.y_p = None
        self.model_c = self.holdout_c = None
        self.model_p = self.holdout_p = None

    def on_command(self, msg: String):
        parts = msg.data.strip().lower().split()
        cmd, args = parts[0], parts[1:]

        # fruit
        if cmd == '/load' and args == ['fruits']:
            self.df_f, self.X_f, self.y_f = fruit_load()
            self.get_logger().info(f"[fruits] Loaded {len(self.X_f)} samples x {self.X_f.shape[1]} features")

        elif cmd == '/visualise' and args == ['fruits']:
            path = fruit_visualize(self.df_f)
            self.get_logger().info(f"[fruits] Visualization saved to: {path}")

        elif cmd == '/train' and args == ['fruits']:
            self.model_f, self.holdout_f = fruit_train(self.X_f, self.y_f)
            self.get_logger().info("[fruits] Logistic Regression trained")

        elif cmd == '/eval' and args == ['fruits']:
            metrics, path = fruit_evaluate(self.model_f, self.holdout_f)
            self.get_logger().info(f"[fruits] Accuracy: {metrics['accuracy']:.3f}")
            self.get_logger().info(f"[fruits] Report:\n{metrics['report']}")
            self.get_logger().info(f"[fruits] Evaluation image: {path}")

        elif cmd == '/importance' and args == ['fruits']:
            path = fruit_importance(self.model_f, self.X_f)
            self.get_logger().info(f"[fruits] Feature importances saved to: {path}")

        # iris
        elif cmd == '/load' and args == ['iris']:
            self.df_i, self.X_i, self.y_i = iris_load()
            self.get_logger().info(f"[iris] Loaded {len(self.X_i)} samples x {self.X_i.shape[1]} features")

        elif cmd == '/visualise' and args == ['iris']:
            path = iris_visualize(self.df_i)
            self.get_logger().info(f"[iris] Visualization saved to: {path}")

        elif cmd == '/train' and len(args) == 2 and args[0] == 'iris':
            model_key = args[1]
            self.model_i, self.holdout_i = iris_train(self.X_i, self.y_i, model_key)
            self.get_logger().info(f"[iris] {model_key} classifier trained")

        elif cmd == '/eval' and len(args) == 2 and args[0] == 'iris':
            model_key = args[1]
            metrics, path = iris_evaluate(self.model_i, self.holdout_i)
            self.get_logger().info(f"[iris-{model_key}] Accuracy: {metrics['accuracy']:.3f}")
            self.get_logger().info(f"[iris-{model_key}] Report:\n{metrics['report']}")
            self.get_logger().info(f"[iris-{model_key}] Evaluation image: {path}")

        elif cmd == '/importance' and len(args) == 2 and args[0] == 'iris':
            model_key = args[1]
            path = iris_importance(self.model_i, self.X_i, self.y_i)
            self.get_logger().info(f"[iris-{model_key}] Feature importances saved to: {path}")

        # cancer & penguin
        elif cmd == '/load' and args == ['cancer_penguin']:
            self.df_c, self.X_c, self.y_c = load_and_featurize_cancer()
            self.df_p, self.X_p, self.y_p = load_and_featurize_penguin()
            self.get_logger().info(f"[cancer_penguin] Loaded cancer: {len(self.X_c)}; penguin: {len(self.X_p)} samples")

        elif cmd == '/visualise' and args == ['cancer_penguin']:
            path = cp_visualize(self.df_c, self.df_p)
            self.get_logger().info(f"[cancer_penguin] Visualization saved to: {path}")

        elif cmd == '/train' and args == ['cancer_penguin']:
            self.model_c, self.holdout_c = cp_train_cancer(self.X_c, self.y_c)
            self.model_p, self.holdout_p = cp_train_penguin(self.X_p, self.y_p)
            self.get_logger().info("[cancer_penguin] Linear SVMs trained for both datasets")

        elif cmd == '/eval' and args == ['cancer_penguin']:
            metrics, path = cp_evaluate_both(self.model_c, self.holdout_c, self.model_p, self.holdout_p)
            self.get_logger().info(f"[cancer_penguin] Cancer acc: {metrics['cancer_acc']:.3f}")
            self.get_logger().info(f"[cancer_penguin] Penguin acc: {metrics['penguin_acc']:.3f}")
            self.get_logger().info(f"[cancer_penguin] Evaluation image: {path}")

        elif cmd == '/importance' and args == ['cancer_penguin']:
            path = cp_importance_both(self.model_c, self.X_c, self.model_p, self.X_p)
            self.get_logger().info(f"[cancer_penguin] Feature importances saved to: {path}")

        else:
            self.get_logger().warn(f"Unknown or invalid command: '{msg.data}'")


def main():
    rclpy.init()
    node = MLControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
