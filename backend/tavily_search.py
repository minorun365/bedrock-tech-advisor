import os
import json
from tavily import TavilyClient

def lambda_handler(event, context):
    # 環境変数からAPIキーを取得（簡易用。可能ならSecrets Manager等を使いましょう）
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    
    # eventからクエリパラメータを取得
    parameters = event.get('parameters', [])
    for param in parameters:
        if param.get('name') == 'query':
            query = param.get('value')
            break
    
    # Tavilyクライアントを初期化して検索を実行
    client = TavilyClient(api_key=tavily_api_key)
    search_result = client.get_search_context(
        query=query,
        search_depth="advanced",
        max_results=10
    )
    
    # 成功レスポンスを返す
    return {
        'messageVersion': event['messageVersion'],
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': json.dumps(search_result, ensure_ascii=False)
                    }
                }
            }
        }
    }