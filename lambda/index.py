# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
import urllib.request

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-2b-jpn-it")
API_URL = "https://a814-35-196-164-95.ngrok-free.app/".rstrip('/')
SYSTEM_PROMPT = (
    "以下は会話の履歴です。あくまでコンテキストの参照用です。\n"
    "これらを出力に含めず、必ず「アシスタント: 」以降の応答のみを返してください。\n"
)

def invoke_model(payload: dict) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{API_URL}/generate",
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise Exception(f"API error: {resp.status}")
        result = json.load(resp)
        print(result)
        return result
    
def build_prompt(messages: list[dict]) -> str:
    lines = []
    lines.append(SYSTEM_PROMPT.rstrip())
    for msg in messages:
        # ロール名を日本語にマッピング
        role = "ユーザ" if msg.get("role") == "user" else "アシスタント"
        # content は [{"text": ...}, …] のリストなので、全ての text をつなげる
        texts = [c.get("text", "") for c in msg.get("content", [])]
        content_str = "".join(texts)
        lines.append(f"{role}: {content_str}")
    # モデルに続きを生成させるために、最後にアシスタントの開始行を追加
    lines.append("アシスタント: ")
    # 改行で結合して一つの文字列に
    return "\n".join(lines)


def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Nova Liteモデル用のリクエストペイロードを構築
        # 会話履歴を含める
        bedrock_messages = build_prompt(messages)

        request_payload = {
            "prompt": bedrock_messages,
            "max_new_tokens": 512,
            "temperature": 0.7, 
            "top_p": 0.9, 
            "do_sample": True
        }
        
        print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))
        
        # APIにアクセスし、結果を得る
        result = invoke_model(request_payload)
        
        # アシスタントの応答を取得
        assistant_response = result['generated_text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }

if __name__ == "__main__":
    sample_messages = [
        {
            "role": "user",
            "content": [
                {"text": "おはようございます。今日の天気はどうですか？"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"text": "おはようございます！今日は晴れの予報です。最高気温は25℃くらいですよ。"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"text": "今の気温にちょうど良い食べ物はなんですか"}
            ]
        }
    ]

    message  = build_prompt(sample_messages)
    print(message)

    # payloadの設定
    request_payload = {
        "prompt": message,
        "max_new_tokens": 512,
        "temperature": 0.7, 
        "top_p": 0.9, 
        "do_sample": True
    }

    # APIにアクセス
    result = invoke_model(request_payload)
    print(f"Response: {result['generated_text']}")
    print(f"Model processing time: {result['response_time']:.2f}s")
