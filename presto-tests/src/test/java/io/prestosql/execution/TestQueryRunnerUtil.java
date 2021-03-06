/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.prestosql.execution;

import com.google.common.collect.ImmutableSet;
import io.prestosql.Session;
import io.prestosql.plugin.tpch.TpchPlugin;
import io.prestosql.server.BasicQueryInfo;
import io.prestosql.spi.QueryId;
import io.prestosql.tests.DistributedQueryRunner;

import java.util.Set;

import static io.airlift.concurrent.MoreFutures.getFutureValue;
import static io.prestosql.execution.QueryState.RUNNING;
import static io.prestosql.testing.TestingSession.testSessionBuilder;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

public final class TestQueryRunnerUtil
{
    private TestQueryRunnerUtil() {}

    public static QueryId createQuery(DistributedQueryRunner queryRunner, Session session, String sql)
    {
        QueryManager queryManager = queryRunner.getCoordinator().getQueryManager();
        getFutureValue(queryManager.createQuery(session.getQueryId(), new TestingSessionContext(session), sql));
        return session.getQueryId();
    }

    public static void cancelQuery(DistributedQueryRunner queryRunner, QueryId queryId)
    {
        queryRunner.getCoordinator().getQueryManager().cancelQuery(queryId);
    }

    public static void waitForQueryState(DistributedQueryRunner queryRunner, QueryId queryId, QueryState expectedQueryState)
            throws InterruptedException
    {
        waitForQueryState(queryRunner, queryId, ImmutableSet.of(expectedQueryState));
    }

    public static void waitForQueryState(DistributedQueryRunner queryRunner, QueryId queryId, Set<QueryState> expectedQueryStates)
            throws InterruptedException
    {
        QueryManager queryManager = queryRunner.getCoordinator().getQueryManager();
        do {
            // Heartbeat all the running queries, so they don't die while we're waiting
            for (BasicQueryInfo queryInfo : queryManager.getQueries()) {
                if (queryInfo.getState() == RUNNING) {
                    queryManager.recordHeartbeat(queryInfo.getQueryId());
                }
            }
            MILLISECONDS.sleep(500);
        }
        while (!expectedQueryStates.contains(queryManager.getQueryState(queryId)));
    }

    public static DistributedQueryRunner createQueryRunner()
            throws Exception
    {
        DistributedQueryRunner queryRunner = DistributedQueryRunner.builder(testSessionBuilder().build())
                .setNodeCount(2)
                .build();

        try {
            queryRunner.installPlugin(new TpchPlugin());
            queryRunner.createCatalog("tpch", "tpch");
            return queryRunner;
        }
        catch (Exception e) {
            queryRunner.close();
            throw e;
        }
    }
}
