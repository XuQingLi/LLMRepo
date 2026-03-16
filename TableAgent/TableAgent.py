import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent


# ==========================
# 1 读取环境变量
# ==========================

env_path = "/t9k/mnt/lxq/TableAgent/.env"
load_dotenv(env_path)

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

DATA_PATH = "/t9k/mnt/lxq/TableAgent/titanic_cleaned.csv"


# ==========================
# 2 初始化 LLM
# ==========================

llm = ChatOpenAI(
    model=LLM_MODEL_ID,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0
)


# ==========================
# 3 Tool：数据统计
# ==========================

@tool
def data_summary() -> str:
    """生成Titanic数据的summary并保存为CSV文件"""

    import pandas as pd

    df = pd.read_csv(DATA_PATH)

    # 只统计数值列
    summary = df.describe().T

    summary_path = "/t9k/mnt/lxq/TableAgent/summary.csv"

    summary.to_csv(summary_path)

    return f"summary文件已生成: {summary_path}"


# ==========================
# 4 Tool：画图
# ==========================

@tool
def plot_survived_distribution() -> str:
    """绘制Survived分布图"""

    import matplotlib.pyplot as plt

    df = pd.read_csv(DATA_PATH)

    plt.figure()

    df["Survived"].value_counts().plot(kind="bar")

    plt.title("Survived Distribution")
    plt.xlabel("Survived")
    plt.ylabel("Count")

    save_path = "/t9k/mnt/lxq/TableAgent/survived_distribution.png"

    plt.savefig(save_path)

    return f"图像已保存: {save_path}"


# ==========================
# 5 Tool：机器学习
# ==========================

@tool
def train_survival_model() -> str:
    """训练sklearn模型预测Survived"""

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # 自动处理字符串特征
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return f"模型训练完成 Accuracy={acc}"


# ==========================
# 6 注册 Tools
# ==========================

tools = [
    data_summary,
    plot_survived_distribution,
    train_survival_model
]


# ==========================
# 7 创建 Agent（LangChain 1.x）
# ==========================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个数据分析助手，可以调用工具分析CSV数据"
)


# ==========================
# 8 运行任务
# ==========================

if __name__ == "__main__":

    task = """
    请完成以下任务：

    1 统计Titanic数据summary（均值、方差、最大最小值）
    2 绘制Survived分布图
    3 训练sklearn模型预测Survived

    数据路径:
/t9k/mnt/lxq/TableAgent/titanic_cleaned.csv
    """

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )

    print("\n====== Agent Result ======\n")
    print(result)