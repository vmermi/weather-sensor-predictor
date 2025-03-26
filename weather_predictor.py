from typing import Sequence, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import History, EarlyStopping
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import pandas as pd


class WeatherPredictor(Sequential):
    """
    A model that can be used to create a neural network and train on data. The model
    uses the tensorflow.keras API, therefore most arguments have the same options (e.g.
    loss, optimizer, etc.).

    Args:
        x (np.ndarray): Input data
        y (np.ndarray): Output data
        scale_params (pd.DataFrame): Parameters that can be used to denormalize data
            back into their original scale
        neurons (Sequence[int] | None, optional): The amount of neurons per layer. It
            should contain a Sequence of numbers; each item corresponding to a layer of
            this number of neurons. Defaults to [32, 32].
        activation (str | Sequence[str], optional): The activation function of the
            hidden layers. If a sequence is provided, it should have the same length as
            the layers, so that different functions can be used per layer. If a string,
            the same activation function will be used for all layers. Defaults to
            "relu".
        loss (str, optional): The loss function used to quantify the efficiency of the
            model. Defaults to "mae".
        optimizer (str, optional): The optimizer used to adjust the weights of the
        neural network. Defaults to "SGD".
        classes (Sequence | None, optional): If not None, this sequence will be used
            to stratify the validation split of the data. Defaults to None.
        validation_split (float, optional): The percentage of the data used for
            validation. Defaults to 0.2.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scale_params: pd.DataFrame,
        neurons: Sequence[int] | None = None,
        activation: str | Sequence[str] = "relu",
        loss: str = "mae",
        optimizer: str = "SGD",
        classes: Sequence | None = None,
        validation_split: float = 0.2,
    ):
        if neurons is None:
            neurons = [32, 32]
        if (
            isinstance(activation, Sequence)
            and not isinstance(activation, str)
            and len(neurons) != len(activation)
        ):
            raise ValueError(
                "If activation is an Sequence, it must have the same length as neurons"
            )

        super().__init__()
        self.add(Input((x.shape[-1],)))
        if isinstance(activation, Sequence) and not isinstance(activation, str):
            for n, a in zip(neurons, activation):
                self.add(Dense(n, activation=a))
        else:
            for n in neurons:
                self.add(Dense(n, activation=activation))
        self.add(Dense(y.shape[-1], activation="linear"))

        if classes is None:
            (
                self.x_train,
                self.x_val,
                self.y_train,
                self.y_val,
                self.scale_params_train,
                self.scale_params_val,
            ) = train_test_split(x, y, scale_params, test_size=validation_split)
        else:
            (
                self.x_train,
                self.x_val,
                self.y_train,
                self.y_val,
                self.scale_params_train,
                self.scale_params_val,
            ) = train_test_split(
                x, y, scale_params, test_size=validation_split, stratify=classes
            )

        self.callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
        ]

        self.compile(
            optimizer=optimizer,
            loss=loss,
        )

    def fit(self, epochs: int = 100, **kwargs: Any) -> Any:
        """
        A method that uses the training/validation data defined on instanciation to fit,
        using the parent's class fit method.

        Args:
            epochs (int, optional): The maximum amount of epochs to train. Early
                stopping will be used to prevent overfitting. Defaults to 100.

        Returns:
            Any: Returns the History provided by the parent's fit method
        """
        return super().fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=epochs,
            callbacks=self.callbacks,
            **kwargs
        )
