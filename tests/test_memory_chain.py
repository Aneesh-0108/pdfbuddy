from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from memory_chain import create_rag_chain


class TestMemoryChain(TestCase):
    def test_create_rag_chain_uses_supplied_retriever(self):
        mock_retriever = MagicMock(name="retriever")
        mock_history_retriever = MagicMock(name="history_retriever")
        mock_qa_chain = MagicMock(name="qa_chain")
        mock_rag_chain = MagicMock(name="rag_chain")

        with patch(
            "memory_chain.create_history_aware_retriever",
            return_value=mock_history_retriever,
        ) as mock_history, patch(
            "memory_chain.create_stuff_documents_chain", return_value=mock_qa_chain
        ) as mock_stuff, patch(
            "memory_chain.create_retrieval_chain", return_value=mock_rag_chain
        ) as mock_chain:
            result = create_rag_chain(mock_retriever)

        self.assertIs(result, mock_rag_chain)
        mock_history.assert_called_once()
        mock_stuff.assert_called_once()
        mock_chain.assert_called_once_with(mock_history_retriever, mock_qa_chain)


if __name__ == "__main__":
    main()