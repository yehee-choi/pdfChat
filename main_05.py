import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Streamlit 페이지 설정
st.set_page_config(
    page_title="PDF RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 세션 상태 초기화
if 'chromadb' not in st.session_state:
    st.session_state.chromadb = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def load_and_process_pdf(uploaded_file):
    """PDF 파일을 로드하고 처리하는 함수"""
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # PDF 로더 설정
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # 텍스트 분리
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        texts = text_splitter.split_documents(pages)
        
        # OpenAI Embeddings 모델 로드
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # ChromaDB 생성
        chromadb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings_model,
            collection_name='pdf_documents'
        )
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        return chromadb, len(texts)
        
    except Exception as e:
        st.error(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")
        return None, 0

def create_lcel_chain(chromadb):
    """LCEL 체인 생성"""
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    
    prompt = PromptTemplate.from_template(
        """주어진 문서를 바탕으로 질문에 답변해주세요.
        
문서: {context}
질문: {question}

답변은 한국어로, 문서의 내용을 바탕으로 정확하고 자세하게 해주세요.
답변:"""
    )
    
    lcel_chain = (
        {"context": chromadb.as_retriever(search_kwargs={"k": 3}), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return lcel_chain

def get_answer(question, chromadb):
    """질문에 대한 답변을 생성하는 함수"""
    try:
        lcel_chain = create_lcel_chain(chromadb)
        result = lcel_chain.invoke(question)
        return result
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# 메인 앱
def main():
    st.title("📚 PDF RAG 챗봇")
    st.markdown("PDF 파일을 업로드하고 문서 내용에 대해 질문해보세요!")
    
    # 사이드바 - PDF 업로드
    with st.sidebar:
        st.header("📤 PDF 업로드")
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="PDF 파일을 업로드하면 문서 내용을 분석합니다."
        )
        
        if uploaded_file is not None:
            if st.button("📊 문서 처리하기"):
                with st.spinner("PDF 파일을 처리하고 있습니다..."):
                    chromadb, chunk_count = load_and_process_pdf(uploaded_file)
                    
                    if chromadb:
                        st.session_state.chromadb = chromadb
                        st.session_state.pdf_processed = True
                        st.success(f"✅ PDF 처리 완료!")
                        st.info(f"📄 총 {chunk_count}개의 텍스트 청크로 분할되었습니다.")
                    else:
                        st.session_state.pdf_processed = False
        
        # 처리 상태 표시
        if st.session_state.pdf_processed:
            st.success("🟢 PDF 처리 완료")
        else:
            st.warning("🟡 PDF를 업로드해주세요")
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 챗봇")
        
        # PDF가 처리되지 않은 경우
        if not st.session_state.pdf_processed:
            st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드하고 처리해주세요.")
            st.markdown("""
            ### 사용 방법:
            1. 사이드바에서 PDF 파일을 업로드
            2. "문서 처리하기" 버튼 클릭
            3. 처리 완료 후 질문 입력
            """)
        else:
            # 채팅 히스토리 표시
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**🙋‍♂️ 질문:** {q}")
                    st.markdown(f"**🤖 답변:** {a}")
                    st.divider()
            
            # 질문 입력
            question = st.text_input(
                "질문을 입력하세요:",
                placeholder="예: 이 문서의 주요 내용은 무엇인가요?",
                key="question_input"
            )
            
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                ask_button = st.button("🔍 질문하기", type="primary")
            with col_btn2:
                if st.button("🗑️ 채팅 기록 삭제"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # 질문 처리
            if ask_button and question.strip():
                with st.spinner("답변을 생성하고 있습니다..."):
                    answer = get_answer(question, st.session_state.chromadb)
                    st.session_state.chat_history.append((question, answer))
                    st.rerun()
            elif ask_button and not question.strip():
                st.warning("질문을 입력해주세요!")
    
    with col2:
        st.header("ℹ️ 정보")
        
        if st.session_state.pdf_processed:
            st.success("📄 PDF 분석 완료")
            st.markdown("**✨ 기능:**")
            st.markdown("- 문서 내용 검색")
            st.markdown("- 질의응답")
            st.markdown("- 다중 쿼리 검색")
        else:
            st.info("PDF 업로드 대기중")
        
        st.markdown("---")
        st.markdown("**🔧 사용된 기술:**")
        st.markdown("- LangChain")
        st.markdown("- OpenAI GPT-3.5")
        st.markdown("- ChromaDB")
        st.markdown("- Streamlit")
        
        # 예시 질문들
        if st.session_state.pdf_processed:
            st.markdown("---")
            st.markdown("**💡 예시 질문:**")
            example_questions = [
                "문서의 주요 내용을 요약해주세요",
                "가장 중요한 키워드는 무엇인가요?",
                "특정 개념에 대해 설명해주세요",
                "문서에서 언급된 수치나 데이터는?",
                "결론이나 요점은 무엇인가요?"
            ]
            
            for eq in example_questions:
                if st.button(f"📝 {eq}", key=f"example_{hash(eq)}", help="클릭하면 질문이 입력됩니다"):
                    st.session_state.question_input = eq

if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.stop()
    
    main()