package com.minicheck.milvus.config;

import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MilvusConfig {

    @Value("${milvus.host}")
    private String host;

    @Value("${milvus.port}")
    private int port;

    private MilvusServiceClient milvusClient;

    @Bean
    public MilvusServiceClient milvusServiceClient() {
        ConnectParam connectParam = ConnectParam.newBuilder()
                .withHost(host)
                .withPort(port)
                .build();

        milvusClient = new MilvusServiceClient(connectParam);
        log.info("Milvus client connected to {}:{}", host, port);
        return milvusClient;
    }

    @PreDestroy
    public void close() {
        if (milvusClient != null) {
            milvusClient.close();
            log.info("Milvus client connection closed");
        }
    }
}
