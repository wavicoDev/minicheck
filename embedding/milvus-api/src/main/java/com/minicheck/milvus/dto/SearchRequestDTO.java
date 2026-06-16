package com.minicheck.milvus.dto;

import jakarta.validation.constraints.Min;
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
public class SearchRequestDTO {

    @NotNull(message = "Query embedding is required")
    private List<Float> queryEmbedding;

    @Min(value = 1, message = "Top K must be at least 1")
    @Builder.Default
    private int topK = 10;

    private String filter;
}
