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
package io.trino.operator.table.json;

import io.trino.spi.ptf.ConnectorTableFunctionHandle;
import io.trino.spi.type.Type;

import static java.util.Objects.requireNonNull;

public class JsonTable
{
    // TODO name with $ - hidden

    /**
     * The function expects the context item (JSON input) in the channel 0, and the parameters row in the channel 1.
     * Other channels in the input page correspond to default values for the value columns. Each value column in the processingPlan knows the indexes of its default channels.
     * This class comprises all information necessary to execute the json_table function:
     *
     * @param parametersRowType type of the row containing the JSON path parameters for the root JSON path
     * @param processingPlan the root of the processing plan tree
     * @param outer the parent-child relationship between the input relation and the processingPlan result
     * @param errorOnError the error behavior: true for ERROR ON ERROR, false for EMPTY ON ERROR
     */
    public record JsonTableFunctionHandle(Type parametersRowType, JsonTablePlanNode processingPlan, boolean outer, boolean errorOnError)
            implements ConnectorTableFunctionHandle
    {
        // input types?
        // output types?

        public JsonTableFunctionHandle
        {
            requireNonNull(parametersRowType, "parametersRowType is null");
            requireNonNull(processingPlan, "processingPlan is null");
        }
    }
}
