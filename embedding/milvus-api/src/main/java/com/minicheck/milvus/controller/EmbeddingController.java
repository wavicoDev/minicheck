package com.minicheck.milvus.controller;

import com.minicheck.milvus.dto.ApiResponseDTO;
import com.minicheck.milvus.dto.EmbeddingRequestDTO;
import com.minicheck.milvus.dto.EmbeddingResponseDTO;
import com.minicheck.milvus.dto.SearchRequestDTO;
import com.minicheck.milvus.service.EmbeddingService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/embeddings")
@RequiredArgsConstructor
public class EmbeddingController {

    private final EmbeddingService embeddingService;

    @PostMapping
    public ResponseEntity<ApiResponseDTO<Long>> insert(@Valid @RequestBody EmbeddingRequestDTO request) {
        Long id = embeddingService.insert(request);
        return ResponseEntity.ok(ApiResponseDTO.success("Embedding inserted successfully", id));
    }

    @PostMapping("/batch")
    public ResponseEntity<ApiResponseDTO<List<Long>>> insertBatch(@Valid @RequestBody List<EmbeddingRequestDTO> requests) {
        List<Long> ids = embeddingService.insertBatch(requests);
        return ResponseEntity.ok(ApiResponseDTO.success("Batch insert successful", ids));
    }

    @PostMapping("/search")
    public ResponseEntity<ApiResponseDTO<List<EmbeddingResponseDTO>>> search(@Valid @RequestBody SearchRequestDTO request) {
        List<EmbeddingResponseDTO> results = embeddingService.search(request);
        return ResponseEntity.ok(ApiResponseDTO.success(results));
    }

    @DeleteMapping("/{documentId}")
    public ResponseEntity<ApiResponseDTO<Void>> delete(@PathVariable String documentId) {
        embeddingService.deleteByDocumentId(documentId);
        return ResponseEntity.ok(ApiResponseDTO.success("Document deleted successfully", null));
    }

    @PostMapping("/collection/init")
    public ResponseEntity<ApiResponseDTO<Void>> initCollection() {
        embeddingService.createCollectionIfNotExists();
        return ResponseEntity.ok(ApiResponseDTO.success("Collection initialized", null));
    }

    @DeleteMapping("/collection")
    public ResponseEntity<ApiResponseDTO<Void>> dropCollection() {
        embeddingService.dropCollection();
        return ResponseEntity.ok(ApiResponseDTO.success("Collection dropped", null));
    }
}
