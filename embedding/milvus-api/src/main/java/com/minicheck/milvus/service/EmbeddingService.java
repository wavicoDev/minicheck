package com.minicheck.milvus.service;

import com.minicheck.milvus.dto.EmbeddingRequestDTO;
import com.minicheck.milvus.dto.EmbeddingResponseDTO;
import com.minicheck.milvus.dto.SearchRequestDTO;
import com.minicheck.milvus.mapper.EmbeddingMapper;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.grpc.MutationResult;
import io.milvus.grpc.SearchResults;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.R;
import io.milvus.param.RpcStatus;
import io.milvus.param.collection.CreateCollectionParam;
import io.milvus.param.collection.DropCollectionParam;
import io.milvus.param.collection.FieldType;
import io.milvus.param.collection.HasCollectionParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.DeleteParam;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.index.CreateIndexParam;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmbeddingService {

    private final MilvusServiceClient milvusClient;
    private final EmbeddingMapper embeddingMapper;

    @Value("${milvus.collection-name}")
    private String collectionName;

    @Value("${milvus.dimension}")
    private int dimension;

    @PostConstruct
    public void init() {
        createCollectionIfNotExists();
    }

    public void createCollectionIfNotExists() {
        R<Boolean> hasCollection = milvusClient.hasCollection(
                HasCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );

        if (hasCollection.getData()) {
            log.info("Collection '{}' already exists", collectionName);
            loadCollection();
            return;
        }

        FieldType idField = FieldType.newBuilder()
                .withName("id")
                .withDataType(DataType.Int64)
                .withPrimaryKey(true)
                .withAutoID(true)
                .build();

        FieldType documentIdField = FieldType.newBuilder()
                .withName("document_id")
                .withDataType(DataType.VarChar)
                .withMaxLength(256)
                .build();

        FieldType contentField = FieldType.newBuilder()
                .withName("content")
                .withDataType(DataType.VarChar)
                .withMaxLength(65535)
                .build();

        FieldType metadataField = FieldType.newBuilder()
                .withName("metadata")
                .withDataType(DataType.VarChar)
                .withMaxLength(4096)
                .build();

        FieldType embeddingField = FieldType.newBuilder()
                .withName("embedding")
                .withDataType(DataType.FloatVector)
                .withDimension(dimension)
                .build();

        CreateCollectionParam createCollectionParam = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .withDescription("MiniCheck embeddings collection")
                .addFieldType(idField)
                .addFieldType(documentIdField)
                .addFieldType(contentField)
                .addFieldType(metadataField)
                .addFieldType(embeddingField)
                .build();

        R<RpcStatus> response = milvusClient.createCollection(createCollectionParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to create collection: " + response.getMessage());
        }

        log.info("Collection '{}' created successfully", collectionName);
        createIndex();
        loadCollection();
    }

    private void createIndex() {
        CreateIndexParam indexParam = CreateIndexParam.newBuilder()
                .withCollectionName(collectionName)
                .withFieldName("embedding")
                .withIndexType(IndexType.IVF_FLAT)
                .withMetricType(MetricType.COSINE)
                .withExtraParam("{\"nlist\": 1024}")
                .build();

        R<RpcStatus> response = milvusClient.createIndex(indexParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to create index: " + response.getMessage());
        }
        log.info("Index created for collection '{}'", collectionName);
    }

    private void loadCollection() {
        R<RpcStatus> response = milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to load collection: " + response.getMessage());
        }
        log.info("Collection '{}' loaded into memory", collectionName);
    }

    public Long insert(EmbeddingRequestDTO request) {
        List<InsertParam.Field> fields = Arrays.asList(
                new InsertParam.Field("document_id", Collections.singletonList(request.getDocumentId())),
                new InsertParam.Field("content", Collections.singletonList(request.getContent())),
                new InsertParam.Field("metadata", Collections.singletonList(request.getMetadata() != null ? request.getMetadata() : "")),
                new InsertParam.Field("embedding", Collections.singletonList(request.getEmbedding()))
        );

        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build();

        R<MutationResult> response = milvusClient.insert(insertParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to insert: " + response.getMessage());
        }

        Long insertedId = response.getData().getIDs().getIntId().getData(0);
        log.info("Inserted document with ID: {}", insertedId);
        return insertedId;
    }

    public List<Long> insertBatch(List<EmbeddingRequestDTO> requests) {
        List<String> documentIds = requests.stream().map(EmbeddingRequestDTO::getDocumentId).toList();
        List<String> contents = requests.stream().map(EmbeddingRequestDTO::getContent).toList();
        List<String> metadataList = requests.stream()
                .map(r -> r.getMetadata() != null ? r.getMetadata() : "")
                .toList();
        List<List<Float>> embeddings = requests.stream().map(EmbeddingRequestDTO::getEmbedding).toList();

        List<InsertParam.Field> fields = Arrays.asList(
                new InsertParam.Field("document_id", documentIds),
                new InsertParam.Field("content", contents),
                new InsertParam.Field("metadata", metadataList),
                new InsertParam.Field("embedding", embeddings)
        );

        InsertParam insertParam = InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build();

        R<MutationResult> response = milvusClient.insert(insertParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to batch insert: " + response.getMessage());
        }

        List<Long> insertedIds = response.getData().getIDs().getIntId().getDataList();
        log.info("Batch inserted {} documents", insertedIds.size());
        return insertedIds;
    }

    public List<EmbeddingResponseDTO> search(SearchRequestDTO request) {
        SearchParam searchParam = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withMetricType(MetricType.COSINE)
                .withOutFields(Arrays.asList("document_id", "content", "metadata"))
                .withTopK(request.getTopK())
                .withVectors(Collections.singletonList(request.getQueryEmbedding()))
                .withVectorFieldName("embedding")
                .withParams("{\"nprobe\": 10}")
                .withExpr(request.getFilter())
                .build();

        R<SearchResults> response = milvusClient.search(searchParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to search: " + response.getMessage());
        }

        return embeddingMapper.toResponseDTOList(response.getData());
    }

    public void deleteByDocumentId(String documentId) {
        String expr = String.format("document_id == \"%s\"", documentId);

        DeleteParam deleteParam = DeleteParam.newBuilder()
                .withCollectionName(collectionName)
                .withExpr(expr)
                .build();

        R<MutationResult> response = milvusClient.delete(deleteParam);
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to delete: " + response.getMessage());
        }

        log.info("Deleted documents with document_id: {}", documentId);
    }

    public void dropCollection() {
        R<RpcStatus> response = milvusClient.dropCollection(
                DropCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build()
        );
        if (response.getStatus() != R.Status.Success.getCode()) {
            throw new RuntimeException("Failed to drop collection: " + response.getMessage());
        }
        log.info("Collection '{}' dropped", collectionName);
    }
}
