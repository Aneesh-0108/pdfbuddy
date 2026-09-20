from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from indexing import create_vectorstore_from_pdf


class TestIndexing(TestCase):
    def test_create_vectorstore_from_pdf_uses_expected_pipeline(self):
        mock_documents = [MagicMock(name="document")]
        mock_chunks = [MagicMock(name="chunk")]
        mock_vectorstore = MagicMock(name="vectorstore")

        with patch("indexing.PyPDFLoader") as mock_loader, patch(
            "indexing.RecursiveCharacterTextSplitter"
        ) as mock_splitter_class, patch("indexing.HuggingFaceEmbeddings") as mock_embeddings_class, patch(
            "indexing.FAISS.from_documents", return_value=mock_vectorstore
        ) as mock_from_documents:
            mock_loader.return_value.load.return_value = mock_documents
            mock_splitter = mock_splitter_class.return_value
            mock_splitter.split_documents.return_value = mock_chunks
            mock_embeddings = mock_embeddings_class.return_value

            result = create_vectorstore_from_pdf("/tmp/uploaded.pdf")

        self.assertIs(result, mock_vectorstore)
        mock_loader.assert_called_once_with("/tmp/uploaded.pdf")
        mock_splitter_class.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_splitter.split_documents.assert_called_once_with(mock_documents)
        mock_embeddings_class.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_from_documents.assert_called_once_with(mock_chunks, mock_embeddings)


if __name__ == "__main__":
    main()