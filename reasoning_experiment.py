import asyncio
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable
from openai import AsyncOpenAI

MODEL = "gpt-4o"
CLIENT = AsyncOpenAI(max_retries=0)
shift = 0
batch = 256

df_basic = pd.read_csv("./code/basics_with_nouns.csv")[0:batch]


def generate_experiment(column) -> Iterable[Any]:
    if column=='abc':
        df_experiment = df_basic.copy()
        permutation = np.random.permutation(batch)
        df_experiment["permutation"] = permutation
        df_experiment.sort_values(by=['permutation'], inplace=True, ignore_index=True)
    elif column=='nouns':
        df_experiment = pd.read_csv(f"./code/experiment_{shift}_combined.csv")
        permutation = df_experiment['permutation']
    return df_experiment
    

def _format_message(text: str) -> Dict[str, str]:
    return dict(role="user", content=text)


def _pluck_response_content(response) -> str:
    return response.choices[0].message.content


async def ask_gpt(message: str) -> str:
    return _pluck_response_content(
        await CLIENT.chat.completions.create(
            model=MODEL,
            messages=[_format_message(message)]
        ))


def create_prompt(syllogism: Any) -> str:
    return f"{syllogism}. Does the conclusion necessarily follows from the premises? Let's think this through, step by step. The last character of your answer should contain either 0 or 1, for False or True accordingly."


def save_results(results: Iterable[str], df_experiment, column) -> None:
    validation_flags = ["valid" if "1" in answer[-10:] else "invalid" for answer in results]
    df_experiment[f"result - validation flags ({column})"] = validation_flags
    df_experiment.to_csv(f"./code/experiment_{shift}_combined.csv")
    print(f"GPT answers ({column}): \n", results[0:min(batch, 20)])


async def main() -> None:
    for column in ['abc', 'nouns']:
        df_experiment = generate_experiment(column)
        if column=='abc':
            syllogism_list = df_experiment['syllogisms'].tolist()
        elif column=='nouns':
            syllogism_list = df_experiment['syllogisms with nouns'].tolist()    
        requests = (
            ask_gpt(create_prompt(syllogism))
            for syllogism in syllogism_list
        )
        save_results(await asyncio.gather(*requests), df_experiment, column)


if __name__ == "__main__":
    asyncio.run(main())
