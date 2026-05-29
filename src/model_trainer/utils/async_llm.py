import os
import sys
import asyncio
from openai import AsyncOpenAI  


from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Literal, Optional, List
from tenacity import retry, stop_after_attempt, wait_fixed
import asyncio

key="sk-WF4RitT3OZS0ieH5rsHrghv8arWWgoP6v88DFwM7ZeriSx8l"
client = AsyncOpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=key
)
completion_tokens = prompt_tokens = 0
Model = Literal["deepseek-ai/DeepSeek-V3","gpt-4o"]



async def async_get_completion(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int =8192, stop_strs: Optional[List[str]] = None, n=1) -> str:
    global completion_tokens, prompt_tokens
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        n=n,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    if n > 1:
        responses = [choice.text.replace('>', '').strip() for choice in response.choices]
        return responses
    return response.choices[0].text.replace('>', '').strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def async_get_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 2000, stop_strs: Optional[List[str]] = None, messages=None, n=1) -> str:
    global completion_tokens, prompt_tokens
    if messages is None:
        if messages is None:
            messages = [
                # {"role": "system", "content": "You are a helpful assistant analyzing electricity load predictions."},
                {"role": "system", "content": "You are a helpful assistant analyzing bitcoin price."},
                {"role":"user","content":prompt }
            ]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        n=n,
        temperature=temperature,
    )
    
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    if n > 1:
        responses = [choice.message.content.replace('>', '').strip() for choice in response.choices]
        return responses
    return response.choices[0].message.content.replace('>', '').strip()


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
async def async_llm_response(prompt, model: Model, temperature: float = 0.0, max_tokens: int = 8192, stop_strs: Optional[List[str]] = None, n=1) -> str:
    if isinstance(prompt, str):

        content = await async_get_chat(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, n=n)
    else:
        messages = prompt
        prompt = prompt[1]['content']
        content = await async_get_chat(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens, stop_strs=stop_strs, messages=messages, n=n)
    return content

def get_price(model="deepseek-ai/DeepSeek-V3", cached=False):
    """
    根据累积的令牌数计算总费用。
    
    Args:
        model (str): 模型名称，例如 'gpt-4o' 或 'gpt-4o-mini'。
        cached (bool): 是否为缓存令牌（影响单价）。
        
    Returns:
        tuple: (prompt_tokens, completion_tokens, total_cost_in_dollars)
    """
    global completion_tokens, prompt_tokens
    
    # 更新价格表
    pricing = {
        "gpt-4o": {
            "prompt": 2.50 if not cached else 1.25,
            "completion": 10.00 if not cached else 5.00,
        },
        "gpt-4o-mini": {
            "prompt": 0.15 if not cached else 0.075,
            "completion": 0.075,  # mini 版本无缓存区分
        },
        "deepseek-ai/DeepSeek-V3":{
           "prompt": 0.14 if not cached else 0.014,  
           "completion": 0.28,  
        }
    }
    
    # 获取模型价格
    model_pricing = pricing.get(model, {"prompt": 0.0, "completion": 0.0})
    prompt_cost = model_pricing["prompt"]
    completion_cost = model_pricing["completion"]
    
    # 计算总费用
    total_cost = (prompt_tokens * prompt_cost / 1000000) + (completion_tokens * completion_cost / 1000000)
    return prompt_tokens, completion_tokens, total_cost
