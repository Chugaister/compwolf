from asyncio import sleep, CancelledError
from json import dumps
import numpy as np

from .exceptions import *
import db


class Core:

    @staticmethod
    async def jacobi(input_data: list[list[float]]) -> list[float]:
        await sleep(30)
        precision = 0.001
        max_iterations = 1000
        m = np.array(input_data)
        A = m[:, :-1]
        b = m[:, -1:]
        B = np.zeros(A.shape)
        D = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if i == j:
                    B[i, j] = A[i, j]
                else:
                    D[i, j] = A[i, j]
        Binv = np.linalg.inv(B)
        C = np.matmul(Binv, D) * (-1)
        d = np.matmul(Binv, b)
        x = b.copy()
        for i in range(max_iterations):
            x_new = np.matmul(C, x) + d
            if np.linalg.norm(x_new - x) < precision:
                return list(x_new.transpose())
            x = x_new
        else:
            raise MaxIterationsError()

    numerical_methods = {
        "jacobi": jacobi
    }

    @staticmethod
    def validate_input_data(method: str, input_data: list | dict) -> bool:
        match method:
            case "jacobi":
                if not isinstance(input_data, list):
                    return False
                for row in input_data:
                    if not isinstance(row, list):
                        return False
                    for element in row:
                        if not isinstance(element, int) and not isinstance(element, float):
                            return False
                m = len(input_data)
                n = len(input_data[0])
                for row in input_data:
                    if len(row) != n:
                        return False
                if m + 1 != n:
                    return False
                return True

    @staticmethod
    def validate_method(method: str) -> bool:
        return method in Core.numerical_methods.keys()

    @staticmethod
    async def compute(task_id: int):
        session = db.Session()
        task = session.query(db.models.Task).filter_by(id=task_id).first()
        try:
            output_data = await Core.numerical_methods[task.method](task.get_input_data())
        except CancelledError:
            return
        except ComputingError as exc:
            task.status = "error"
            task.error = f"{type(exc).__name__}: {exc}"
            session.commit()
            return
        task.output_data = dumps(output_data)
        task.status = "finished"
        task.progress = 100
        session.commit()
