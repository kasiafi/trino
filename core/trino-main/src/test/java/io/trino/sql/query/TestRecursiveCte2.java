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
package io.trino.sql.query;

import io.trino.Session;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static io.trino.SystemSessionProperties.MAX_RECURSION_DEPTH;
import static io.trino.SystemSessionProperties.getMaxRecursionDepth;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/*
git bisect run bash -xeuc '
  mvnd clean install -am -pl :trino-main -T2C -DskipTests \
    -Dmaven.javadoc.skip=true -Dmaven.source.skip=true -Dair.check.skip-all=true || exit 125
  mvnd test -pl :trino-main -Dair.check.skip-all=true -Dtest=TestRecursiveCte2
'
 */
@TestInstance(PER_CLASS)
public class TestRecursiveCte2
{
    private QueryAssertions assertions;

    @BeforeAll
    public void init()
    {
        assertions = new QueryAssertions();
    }

    @AfterAll
    public void teardown()
    {
        assertions.close();
        assertions = null;
    }

    @Test
    public void testLambda()
    {
        assertThatThrownBy(() -> assertions.query("with recursive gbt(list, status, level) as ( " +
                "select " +
                "array[0] as list, " +
                "true as status, " +
                "0 as level " +
                "union all " +
                "select " +
                "list || gbt.level, " +
                "jq.result, " +
                "gbt.level + 1 " +
                "from gbt " +
                "join (select true as result) as jq " +
                "on jq.result = gbt.status " +
                "and any_match( list, x -> x = x) " +
                "where gbt.level < 10 " +
                ") " +
                "select * from gbt"))
                .isInstanceOf(NullPointerException.class);
    }
}
