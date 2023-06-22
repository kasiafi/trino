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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

@TestInstance(PER_CLASS)
public class TestJsonTable
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
    public void testSimple()
    {
        /*assertThat(assertions.query("""
                SELECT first, last
                FROM (SELECT '{"a" : [1, 2, 3], "b" : [4, 5, 6]}') t(json_col), JSON_TABLE(
                    json_col,
                    'lax $.a'
                    COLUMNS(
                        first BIGINT PATH 'lax $[0]',
                        last BIGINT PATH 'lax $[last]'))
               """))
                .matches("VALUES (BIGINT '1', BIGINT '3')");*/

        assertThat(assertions.query("""
                 SELECT *
                 FROM
                     (SELECT '{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}') t(json_col),
                     JSON_TABLE(
                         json_col,
                         'lax $.a' AS "path_a"
                         COLUMNS(
                             NESTED PATH 'lax $.b[*]' AS "path_b"
                                     COLUMNS (c1 INTEGER PATH 'lax $ * 10'),
                             NESTED PATH 'lax $.c' AS "path_c"
                                     COLUMNS (
                                         NESTED PATH 'lax $[0][*]' AS "path_d" COLUMNS (c2 INTEGER PATH 'lax $ * 100'),
                                         NESTED PATH 'lax $[last][*]' AS "path_e" COLUMNS (c3 INTEGER PATH 'lax $ * 1000')))
                         PLAN ("path_a" OUTER ("path_b" UNION ("path_c" INNER ("path_d" CROSS "path_e")))))
                """))
                .matches("""
                        VALUES
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', 10, CAST(null AS integer), CAST(null AS integer)),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', 20, null, null),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', 30, null, null),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 400, 7000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 400, 8000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 400, 9000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 500, 7000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 500, 8000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 500, 9000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 600, 7000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 600, 8000),
                            ('{"a" : {"b" : [1, 2, 3], "c" : [[4, 5, 6], [7, 8, 9]]}}', null, 600, 9000)
                        """);
    }
}
