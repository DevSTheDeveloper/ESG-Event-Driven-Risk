# ESG-Event-Driven-Risk

**Event-driven modeling of financial risk from ESG incidents**

This project applies machine learning and event-study methodology to quantify the financial impact of ESG-related events. Using [`finbert-esg-9-categories`](https://huggingface.co/yiyanghkust/finbert-esg-9-categories), company disclosures and sustainability reports are classified into nine ESG event categories (e.g., climate change, human capital, governance). Each detected incident is then linked to market data: we pull **minute-level ticker prices from Yahoo Finance**, constructing an event window spanning the day before through the day after the disclosure.

By combining **ESG NLP classification** with **short-horizon abnormal return analysis**, the pipeline measures how specific ESG shocks translate into price reactions and volatility shifts. The framework demonstrates how unstructured disclosures can be systematically converted into structured risk signals—bridging natural language processing with financial time series analysis for practical ESG risk modeling.
