package com.minicheck.milvus.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EmbeddingRequestDTO {

    @NotBlank(message = "Document ID is required")
    private String documentId;

    @NotBlank(message = "Content is required")
    private String content;

    @NotNull(message = "Embedding vector is required")
    private List<Float> embedding;

    private String metadata;
}
