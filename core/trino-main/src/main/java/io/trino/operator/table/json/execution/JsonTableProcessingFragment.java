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
package io.trino.operator.table.json.execution;

import com.fasterxml.jackson.databind.JsonNode;

public interface JsonTableProcessingFragment
{
    /**
     * Prepares the Fragment to produce rows for the new JSON item.
     */
    void reset(JsonNode item);

    /**
     * Tries to produce output values for all columns included in the Fragment,
     * and stores them in corresponding positions in `newRow`.
     * Note: According to OUTER or UNION semantics, some values might be null-padded instead of computed.
     * @return true if row was produced, false if row was not produced (Fragment is finished)
     */
    boolean getRow();

    /**
     * Returns an array containing indexes of columns produced by the fragment within all columns produced by json_table.
     */
    int[] getOutputLayout();
}
