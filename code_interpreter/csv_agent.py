from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Start...")

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        path="../data/episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main()
