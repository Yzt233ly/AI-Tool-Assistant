"""工具定义模块。

这里定义了三个「工具」，也就是 LLM 可以调用的 Python 函数。

关于 @tool 装饰器（这是理解 Tool Calling 的关键）：
    一个普通 Python 函数，加上 @tool 装饰器后，就变成了 LangChain 认识的
    "Tool" 对象。LangChain 会自动读取：
      1. 函数名           -> 变成工具的名字（LLM 用它来指定要调用哪个工具）
      2. 函数的 docstring -> 变成工具的描述（LLM 靠它判断什么时候该用这个工具）
      3. 参数的类型注解    -> 变成参数的说明（LLM 靠它知道要给什么参数、参数类型）
      4. docstring 里 Args 段 -> 进一步补充每个参数的含义

    @tool 把这些信息打包成一份 "工具的说明书"（专业叫 schema），
    之后 bind_tools 会把这份说明书发给 LLM。
"""

import httpx
from langchain_core.tools import tool


# 天气代码 -> 中文描述 的映射表。
# Open-Meteo 返回的 weathercode 是一串数字（WMO 标准），
# 我们把常见的数字翻译成中文，让最终回答更友好。
WEATHER_CODES = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "有雾",
    48: "雾凇",
    51: "毛毛雨（轻）",
    53: "毛毛雨（中）",
    55: "毛毛雨（大）",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨（轻）",
    81: "阵雨（中）",
    82: "阵雨（大）",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
    99: "雷阵雨伴大冰雹",
}


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """查询两种货币之间的实时汇率，也就是 1 单位源货币能兑换多少目标货币。

    Args:
        from_currency: 源货币的三位大写代码，例如 USD、EUR、CNY。
        to_currency: 目标货币的三位大写代码，例如 CNY、JPY、GBP。

    Returns:
        str: 用中文描述汇率结果的文本。
    """
    # 把用户/LLM 传入的货币代码统一转成大写，防止 "usd" 这种小写导致查不到
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # Frankfurter 是一个免费、无需 key 的汇率接口
    # 注意：官方已经迁移到 api.frankfurter.dev/v1，旧域名 api.frankfurter.app 会返回 301 跳转
    url = "https://api.frankfurter.dev/v1/latest"
    # params 会被 httpx 拼到 URL 后面，变成 ?from=USD&to=CNY
    response = httpx.get(url, params={"from": from_currency, "to": to_currency})
    # 如果接口返回 4xx/5xx 错误码，这里会抛出异常（在 main.py 里统一处理）
    response.raise_for_status()

    data = response.json()
    # 响应结构类似：{"base": "USD", "rates": {"CNY": 7.2}}
    rate = data["rates"][to_currency]
    return f"1 {from_currency} 可以兑换 {rate} {to_currency}"


@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气，包括天气状况、气温和风速。

    Args:
        city: 城市名称，例如 北京、上海、东京、伦敦。中英文都可以。

    Returns:
        str: 用中文描述该城市当前天气的文本。
    """
    # 第一步：地理编码 —— 把「城市名」转换成「经纬度」。
    # 因为天气接口只认经纬度，不认识城市名，所以要先多查一次。
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    # count=1 表示只要最匹配的第一个结果；language=zh 让返回的城市名是中文
    geo_response = httpx.get(
        geo_url, params={"name": city, "count": 1, "language": "zh"}
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()

    # 如果没搜到结果，直接返回一句提示，让 LLM 转告用户换一个城市名
    results = geo_data.get("results")
    if not results:
        return f"没有找到城市「{city}」，请换一个更明确的城市名。"

    first_result = results[0]
    latitude = first_result["latitude"]
    longitude = first_result["longitude"]
    city_name = first_result.get("name", city)

    # 第二步：用经纬度查询当前天气
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = httpx.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",  # 只要当前天气，不要未来预报
        },
    )
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    current = weather_data["current_weather"]
    temperature = current["temperature"]
    wind_speed = current["windspeed"]
    weather_code = current["weathercode"]
    weather_desc = WEATHER_CODES.get(weather_code, f"天气代码 {weather_code}")

    return f"{city_name}当前天气：{weather_desc}，气温 {temperature}°C，风速 {wind_speed} km/h"


@tool
def get_random_quote() -> str:
    """获取一条随机的英文名言，包含作者信息。

    Returns:
        str: 名言内容和作者。
    """
    # 这个工具没有参数，正好演示「无参数工具」的用法
    # 用 DummyJSON 提供的免费名言接口（无需 key）
    url = "https://dummyjson.com/quotes/random"
    response = httpx.get(url)
    response.raise_for_status()

    data = response.json()
    # 响应结构：{"quote": "名言内容", "author": "作者"}
    return f"「{data['quote']}」 —— {data['author']}"


# 把所有工具收集到一个列表里。
# config.py 会引用它来做 bind_tools，main.py 会引用它来建立 TOOL_MAP。
ALL_TOOLS = [get_exchange_rate, get_weather, get_random_quote]
