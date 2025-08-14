from typing import List, Dict, Tuple
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import requests
import re
import os

class LLM:
    def __init__(self):
        self.client = GigaChat(
            credentials="ВАШ_ТОКЕН",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            verify_ssl_certs=False,
            temperature=0.5  # Для более детерминированных ответов
        )
        
    def generate(self, prompt: str) -> str:
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return "Извините, произошла ошибка при обработке запроса."

class VectorStore:
    def __init__(self):
        # Загружаем модель для эмбеддингов
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        #self.model = SentenceTransformer('models/paraphrase-multilingual')
        self.db = None
        self.metadata = []
    
    def build_index(self, texts: List[str], metadatas: List[dict]):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.model,  # Исправлено: передаем саму модель
            metadatas=metadatas
        )
        self.metadata = metadatas
        
    def search(self, query_text: str, k: int = 5) -> List[Tuple[str, dict]]:
        query_embedding = self.model.encode([query_text])[0]
        if self.db is None:
            raise ValueError("Индекс не построен. Вызовите build_index() сначала.")
        
        # Возвращаем текст и метаданные
        docs = self.db.similarity_search_by_vector(query_embedding, k=k)
        return [(doc.page_content, doc.metadata) for doc in docs]

class AnswerBot:
    def __init__(self, urls: List[str]):
        self.llm = LLM()
        self.vector_store = VectorStore()
        self.prepare_data(urls)
        
    def prepare_data(self, urls: List[str]):
        all_texts = []
        all_metadatas = []
        
        for url in urls:
            try:
                clean_text = self.parse_url(url)
                if clean_text:
                    chunks = self.chunk_text(clean_text)
                    all_texts.extend(chunks)
                    all_metadatas.extend([{"source": url}] * len(chunks))
                    print(f"Обработан URL: {url} | Чанков: {len(chunks)}")
            except Exception as e:
                print(f"Ошибка обработки {url}: {e}")
        
        if all_texts:
            print("Строим векторный индекс...")
            self.vector_store.build_index(all_texts, all_metadatas)
            print(f"Индекс построен. Всего чанков: {len(all_texts)}")
        else:
            print("Нет данных для построения индекса")
    
    def parse_url(self, url: str) -> str:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Удаляем ненужные элементы
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Извлекаем основной контент
        main_content = soup.find("main") or soup
        text = " ".join(p.get_text() for p in main_content.find_all(["p", "h1", "h2", "h3", "div"]))
        return re.sub(r"\s+", " ", text).strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    def generate_answer(self, question: str) -> str:
        # Шаг 1: Поиск релевантных документов
        relevant_docs = self.vector_store.search(question, k=5)
        
        if not relevant_docs:
            return "Не найдено подходящей информации для ответа на вопрос."
        
        # Шаг 2: Подготовка контекста с источниками
        context_parts = []
        source_mapping = {}  # {url: source_id}
        current_source_id = 1
        
        for doc_text, metadata in relevant_docs:
            url = metadata["source"]
            
            # Создаем маппинг URL -> ID
            if url not in source_mapping:
                source_mapping[url] = current_source_id
                current_source_id += 1
                
            source_id = source_mapping[url]
            context_parts.append(f"{doc_text} [{source_id}]")
        
        context = "\n\n".join(context_parts)
        
        # Шаг 3: Формирование промпта для LLM
        prompt = f"""
        Ты - ассистент компании EORA, отвечающий на вопросы клиентов.
        Ответь на вопрос, используя ТОЛЬКО предоставленный контекст.
        В ответе цитируй источники в квадратных скобках [номер] после каждого утверждения.
        Если информации недостаточно, скажи об этом.
        
        Контекст:
        {context}
        
        Вопрос: {question}
        Ответ:
        """
        
        # Шаг 4: Генерация ответа
        answer = self.llm.generate(prompt)
        
        # Шаг 5: Форматирование источников
        sources_list = "\n".join([f"[{id}] {url}" for url, id in source_mapping.items()])
        
        return f"{answer}\n\nИсточники:\n{sources_list}"

# CLI-интерфейс для взаимодействия
def main():
    # Список URL из задания
    urls = [
        "https://eora.ru/cases/promyshlennaya-bezopasnost",
        "https://eora.ru/cases/lamoda-systema-segmentacii-i-poiska-po-pohozhey-odezhde",
        "https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/karas-golosovoy-assistent",
        "https://eora.ru/cases/assistenty-dlya-gorodov",
        "https://eora.ru/cases/avtomatizaciya-v-promyshlennosti/chemrar-raspoznovanie-molekul",
        "https://eora.ru/cases/zeptolab-skazki-pro-amnyama-dlya-sberbox",
        "https://eora.ru/cases/goosegaming-algoritm-dlya-ocenki-igrokov",
        "https://eora.ru/cases/dodo-pizza-robot-analitik-otzyvov",
        "https://eora.ru/cases/ifarm-nejroset-dlya-ferm",
        "https://eora.ru/cases/zhivibezstraha-navyk-dlya-proverki-rodinok",
        "https://eora.ru/cases/sportrecs-nejroset-operator-sportivnyh-translyacij",
        "https://eora.ru/cases/avon-chat-bot-dlya-zhenshchin",
        "https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/navyk-dlya-proverki-loterejnyh-biletov",
        "https://eora.ru/cases/computer-vision/iss-analiz-foto-avtomobilej",
        "https://eora.ru/cases/purina-master-bot",
        "https://eora.ru/cases/skinclub-algoritm-dlya-ocenki-veroyatnostej",
        "https://eora.ru/cases/skolkovo-chat-bot-dlya-startapov-i-investorov",
        "https://eora.ru/cases/purina-podbor-korma-dlya-sobaki",
        "https://eora.ru/cases/purina-navyk-viktorina",
        "https://eora.ru/cases/dodo-pizza-pilot-po-avtomatizacii-kontakt-centra",
        "https://eora.ru/cases/dodo-pizza-avtomatizaciya-kontakt-centra",
        "https://eora.ru/cases/s7-navyk-dlya-podbora-aviabiletov",
        "https://eora.ru/cases/workeat-whatsapp-bot",
        "https://eora.ru/cases/absolyut-strahovanie-navyk-dlya-raschyota-strahovki",
        "https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto",
        "https://eora.ru/cases/kazanexpress-sistema-rekomendacij-na-sajte",
        "https://eora.ru/cases/intels-proverka-logotipa-na-plagiat",
        "https://eora.ru/cases/karcher-viktorina-s-voprosami-pro-uborku",
        "https://eora.ru/cases/chat-boty/purina-friskies-chat-bot-na-sajte",
        "https://eora.ru/cases/nejroset-segmentaciya-video",
        "https://eora.ru/cases/chat-boty/essa-nejroset-dlya-generacii-rolikov",
        "https://eora.ru/cases/qiwi-poisk-anomalij",
        "https://eora.ru/cases/frisbi-nejroset-dlya-raspoznavaniya-pokazanij-schetchikov",
        "https://eora.ru/cases/skazki-dlya-gugl-assistenta",
        "https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie"
    ]
    
    print("Инициализация AnswerBot...")
    bot = AnswerBot(urls)
    print("Готов к работе. Введите ваш вопрос (или 'exit' для выхода).")
    
    while True:
        question = input("\nВаш вопрос: ").strip()
        if question.lower() == "exit":
            break
            
        if not question:
            continue
            
        print("Обработка запроса...")
        response = bot.generate_answer(question)
        print("\n" + "=" * 80)
        print(response)
        print("=" * 80)

if __name__ == "__main__":
    main()