package com.minicheck.milvus.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EmbeddingResponseDTO {

    private Long id;
    private String documentId;
    private String content;
    private String metadata;
    private Float score;
}
