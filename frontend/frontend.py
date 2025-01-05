import os
import json
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from botocore.eventstream import EventStreamError

def initialize_session():
    """セッションの初期設定を行う"""
    if "client" not in st.session_state:
        st.session_state.client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    
    return st.session_state.client, st.session_state.session_id, st.session_state.messages

def display_chat_history(messages):
    """チャット履歴を表示する"""
    st.title("わが家のAI技術顧問")
    st.text("画面下部のチャットボックスから何でも質問してね！")
    
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

def handle_trace_event(event):
    """トレースイベントの処理を行う"""
    if "orchestrationTrace" not in event["trace"]["trace"]:
        return
    
    trace = event["trace"]["trace"]["orchestrationTrace"]
    
    # 「モデル入力」トレースの表示
    if "modelInvocationInput" in trace:
        with st.expander("🤔 思考中…", expanded=False):
            input_trace = trace["modelInvocationInput"]["text"]
            try:
                st.json(json.loads(input_trace))
            except:
                st.write(input_trace)
    
    # 「モデル出力」トレースの表示
    if "modelInvocationOutput" in trace:
        output_trace = trace["modelInvocationOutput"]["rawResponse"]["content"]
        with st.expander("💡 思考がまとまりました", expanded=False):
            try:
                thinking = json.loads(output_trace)["content"][0]["text"]
                if thinking:
                    st.write(thinking)
                else:
                    st.write(json.loads(output_trace)["content"][0])
            except:
                st.write(output_trace)
    
    # 「根拠」トレースの表示
    if "rationale" in trace:
        with st.expander("✅ 次のアクションを決定しました", expanded=True):
            st.write(trace["rationale"]["text"])
    
    # 「ツール呼び出し」トレースの表示
    if "invocationInput" in trace:
        invocation_type = trace["invocationInput"]["invocationType"]
        
        if invocation_type == "AGENT_COLLABORATOR":
            agent_name = trace["invocationInput"]["agentCollaboratorInvocationInput"]["agentCollaboratorName"]
            with st.expander(f"🤖 サブエージェント「{agent_name}」を呼び出し中…", expanded=True):
                st.write(trace["invocationInput"]["agentCollaboratorInvocationInput"]["input"]["text"])
        
        elif invocation_type == "KNOWLEDGE_BASE":
            with st.expander("📖 ナレッジベースを検索中…", expanded=False):
                st.write(trace["invocationInput"]["knowledgeBaseLookupInput"]["text"])
        
        elif invocation_type == "ACTION_GROUP":
            with st.expander("💻 Lambdaを実行中…", expanded=False):
                st.write(trace['invocationInput']['actionGroupInvocationInput'])
    
    # 「観察」トレースの表示
    if "observation" in trace:
        obs_type = trace["observation"]["type"]
        
        if obs_type == "KNOWLEDGE_BASE":
            with st.expander("🔍 ナレッジベースから検索結果を取得しました", expanded=False):
                st.write(trace["observation"]["knowledgeBaseLookupOutput"]["retrievedReferences"])
        
        elif obs_type == "AGENT_COLLABORATOR":
            agent_name = trace["observation"]["agentCollaboratorInvocationOutput"]["agentCollaboratorName"]
            with st.expander(f"🤖 サブエージェント「{agent_name}」から回答を取得しました", expanded=True):
                st.write(trace["observation"]["agentCollaboratorInvocationOutput"]["output"]["text"])

def invoke_bedrock_agent(client, session_id, prompt):
    """Bedrockエージェントを呼び出す"""
    load_dotenv()
    return client.invoke_agent(
        agentId=os.getenv("AGENT_ID"),
        agentAliasId=os.getenv("AGENT_ALIAS_ID"),
        sessionId=session_id,
        enableTrace=True,
        inputText=prompt,
    )

def handle_agent_response(response, messages):
    """エージェントのレスポンスを処理する"""
    with st.chat_message("assistant"):
        for event in response.get("completion"):
            if "trace" in event:
                handle_trace_event(event)
            
            if "chunk" in event:
                answer = event["chunk"]["bytes"].decode()
                st.write(answer)
                messages.append({"role": "assistant", "text": answer})

def show_error_popup(exeption):
    """エラーポップアップを表示する"""
    if exeption == "dependencyFailedException":
        error_message = "【エラー】ナレッジベースのAurora DBがスリープしていたようです。数秒おいてから、ブラウザをリロードして再度お試しください🙏"
    elif exeption == "throttlingException":
        error_message = "【エラー】Bedrockのモデル負荷が高いようです。1分待ってから、ブラウザをリロードして再度お試しください🙏（改善しない場合は、モデルを変更するか[サービスクォータの引き上げ申請](https://aws.amazon.com/jp/blogs/news/generative-ai-amazon-bedrock-handling-quota-problems/)を実施ください）"
    elif exeption == "internalServerException" or exeption == "badGatewayException" or exeption == "internalServerException":
        error_message = "【エラー】エージェントに何か問題が起こったようです。ブラウザをリロードして再度お試しください🙏"
    st.error(error_message)

def main():
    """メインのアプリケーション処理"""
    client, session_id, messages = initialize_session()
    display_chat_history(messages)
    
    if prompt := st.chat_input("例：画像入り資料を使ったRAGアプリを作るにはどうすればいい？"):
        messages.append({"role": "human", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            response = invoke_bedrock_agent(client, session_id, prompt)
            handle_agent_response(response, messages)
            
        except (EventStreamError, ClientError) as e:
            if "dependencyFailedException" in str(e):
                show_error_popup("dependencyFailedException")
            elif "throttlingException" in str(e):
                show_error_popup("throttlingException")
            else:
                raise e

if __name__ == "__main__":
    main()